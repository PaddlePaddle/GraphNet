import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_: torch.Tensor,
        L_self_modules_encoder_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_input_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_input_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_input_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_input_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_output_prelu_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_output_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mask_generator_modules_output_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decoder_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_input_ = L_input_
        l_self_modules_encoder_parameters_weight_ = (
            L_self_modules_encoder_parameters_weight_
        )
        l_self_modules_mask_generator_modules_input_norm_parameters_weight_ = (
            L_self_modules_mask_generator_modules_input_norm_parameters_weight_
        )
        l_self_modules_mask_generator_modules_input_norm_parameters_bias_ = (
            L_self_modules_mask_generator_modules_input_norm_parameters_bias_
        )
        l_self_modules_mask_generator_modules_input_conv_parameters_weight_ = (
            L_self_modules_mask_generator_modules_input_conv_parameters_weight_
        )
        l_self_modules_mask_generator_modules_input_conv_parameters_bias_ = (
            L_self_modules_mask_generator_modules_input_conv_parameters_bias_
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_1_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_1_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_4_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_4_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_bias_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_weight_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_weight_
        l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_bias_ = L_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_bias_
        l_self_modules_mask_generator_modules_output_prelu_parameters_weight_ = (
            L_self_modules_mask_generator_modules_output_prelu_parameters_weight_
        )
        l_self_modules_mask_generator_modules_output_conv_parameters_weight_ = (
            L_self_modules_mask_generator_modules_output_conv_parameters_weight_
        )
        l_self_modules_mask_generator_modules_output_conv_parameters_bias_ = (
            L_self_modules_mask_generator_modules_output_conv_parameters_bias_
        )
        l_self_modules_decoder_parameters_weight_ = (
            L_self_modules_decoder_parameters_weight_
        )
        feats = torch.conv1d(
            l_input_,
            l_self_modules_encoder_parameters_weight_,
            None,
            (8,),
            (8,),
            (1,),
            1,
        )
        l_input_ = l_self_modules_encoder_parameters_weight_ = None
        feats_1 = torch.nn.functional.group_norm(
            feats,
            1,
            l_self_modules_mask_generator_modules_input_norm_parameters_weight_,
            l_self_modules_mask_generator_modules_input_norm_parameters_bias_,
            1e-08,
        )
        l_self_modules_mask_generator_modules_input_norm_parameters_weight_ = (
            l_self_modules_mask_generator_modules_input_norm_parameters_bias_
        ) = None
        feats_2 = torch.conv1d(
            feats_1,
            l_self_modules_mask_generator_modules_input_conv_parameters_weight_,
            l_self_modules_mask_generator_modules_input_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        feats_1 = (
            l_self_modules_mask_generator_modules_input_conv_parameters_weight_
        ) = l_self_modules_mask_generator_modules_input_conv_parameters_bias_ = None
        input_1 = torch.conv1d(
            feats_2,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_2 = torch.prelu(
            input_1,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_1 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_3 = torch.nn.functional.group_norm(
            input_2,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_2 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_4 = torch.conv1d(
            input_3,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (1,),
            (1,),
            512,
        )
        input_3 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_5 = torch.prelu(
            input_4,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_4 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_6 = torch.nn.functional.group_norm(
            input_5,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_5 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual = torch.conv1d(
            input_6,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_res_out_parameters_bias_ = (None)
        skip_out = torch.conv1d(
            input_6,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_6 = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_0_modules_skip_out_parameters_bias_ = (None)
        feats_3 = feats_2 + residual
        feats_2 = residual = None
        output = 0.0 + skip_out
        skip_out = None
        input_7 = torch.conv1d(
            feats_3,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_8 = torch.prelu(
            input_7,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_7 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_9 = torch.nn.functional.group_norm(
            input_8,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_8 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_10 = torch.conv1d(
            input_9,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (2,),
            (2,),
            512,
        )
        input_9 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_11 = torch.prelu(
            input_10,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_10 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_12 = torch.nn.functional.group_norm(
            input_11,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_11 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_1 = torch.conv1d(
            input_12,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_res_out_parameters_bias_ = (None)
        skip_out_1 = torch.conv1d(
            input_12,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_12 = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_1_modules_skip_out_parameters_bias_ = (None)
        feats_4 = feats_3 + residual_1
        feats_3 = residual_1 = None
        output_1 = output + skip_out_1
        output = skip_out_1 = None
        input_13 = torch.conv1d(
            feats_4,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_14 = torch.prelu(
            input_13,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_13 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_15 = torch.nn.functional.group_norm(
            input_14,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_14 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_16 = torch.conv1d(
            input_15,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (4,),
            (4,),
            512,
        )
        input_15 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_17 = torch.prelu(
            input_16,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_16 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_18 = torch.nn.functional.group_norm(
            input_17,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_17 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_2 = torch.conv1d(
            input_18,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_res_out_parameters_bias_ = (None)
        skip_out_2 = torch.conv1d(
            input_18,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_18 = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_2_modules_skip_out_parameters_bias_ = (None)
        feats_5 = feats_4 + residual_2
        feats_4 = residual_2 = None
        output_2 = output_1 + skip_out_2
        output_1 = skip_out_2 = None
        input_19 = torch.conv1d(
            feats_5,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_20 = torch.prelu(
            input_19,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_19 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_21 = torch.nn.functional.group_norm(
            input_20,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_20 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_22 = torch.conv1d(
            input_21,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (8,),
            (8,),
            512,
        )
        input_21 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_23 = torch.prelu(
            input_22,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_22 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_24 = torch.nn.functional.group_norm(
            input_23,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_23 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_3 = torch.conv1d(
            input_24,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_res_out_parameters_bias_ = (None)
        skip_out_3 = torch.conv1d(
            input_24,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_24 = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_3_modules_skip_out_parameters_bias_ = (None)
        feats_6 = feats_5 + residual_3
        feats_5 = residual_3 = None
        output_3 = output_2 + skip_out_3
        output_2 = skip_out_3 = None
        input_25 = torch.conv1d(
            feats_6,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_26 = torch.prelu(
            input_25,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_25 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_27 = torch.nn.functional.group_norm(
            input_26,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_26 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_28 = torch.conv1d(
            input_27,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (16,),
            (16,),
            512,
        )
        input_27 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_29 = torch.prelu(
            input_28,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_28 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_30 = torch.nn.functional.group_norm(
            input_29,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_29 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_4 = torch.conv1d(
            input_30,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_res_out_parameters_bias_ = (None)
        skip_out_4 = torch.conv1d(
            input_30,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_30 = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_4_modules_skip_out_parameters_bias_ = (None)
        feats_7 = feats_6 + residual_4
        feats_6 = residual_4 = None
        output_4 = output_3 + skip_out_4
        output_3 = skip_out_4 = None
        input_31 = torch.conv1d(
            feats_7,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_32 = torch.prelu(
            input_31,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_31 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_33 = torch.nn.functional.group_norm(
            input_32,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_32 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_34 = torch.conv1d(
            input_33,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (32,),
            (32,),
            512,
        )
        input_33 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_35 = torch.prelu(
            input_34,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_34 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_36 = torch.nn.functional.group_norm(
            input_35,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_35 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_5 = torch.conv1d(
            input_36,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_res_out_parameters_bias_ = (None)
        skip_out_5 = torch.conv1d(
            input_36,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_36 = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_5_modules_skip_out_parameters_bias_ = (None)
        feats_8 = feats_7 + residual_5
        feats_7 = residual_5 = None
        output_5 = output_4 + skip_out_5
        output_4 = skip_out_5 = None
        input_37 = torch.conv1d(
            feats_8,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_38 = torch.prelu(
            input_37,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_37 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_39 = torch.nn.functional.group_norm(
            input_38,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_38 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_40 = torch.conv1d(
            input_39,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (64,),
            (64,),
            512,
        )
        input_39 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_41 = torch.prelu(
            input_40,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_40 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_42 = torch.nn.functional.group_norm(
            input_41,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_41 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_6 = torch.conv1d(
            input_42,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_res_out_parameters_bias_ = (None)
        skip_out_6 = torch.conv1d(
            input_42,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_42 = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_6_modules_skip_out_parameters_bias_ = (None)
        feats_9 = feats_8 + residual_6
        feats_8 = residual_6 = None
        output_6 = output_5 + skip_out_6
        output_5 = skip_out_6 = None
        input_43 = torch.conv1d(
            feats_9,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_44 = torch.prelu(
            input_43,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_43 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_45 = torch.nn.functional.group_norm(
            input_44,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_44 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_46 = torch.conv1d(
            input_45,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (128,),
            (128,),
            512,
        )
        input_45 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_47 = torch.prelu(
            input_46,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_46 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_48 = torch.nn.functional.group_norm(
            input_47,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_47 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_7 = torch.conv1d(
            input_48,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_res_out_parameters_bias_ = (None)
        skip_out_7 = torch.conv1d(
            input_48,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_48 = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_7_modules_skip_out_parameters_bias_ = (None)
        feats_10 = feats_9 + residual_7
        feats_9 = residual_7 = None
        output_7 = output_6 + skip_out_7
        output_6 = skip_out_7 = None
        input_49 = torch.conv1d(
            feats_10,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_50 = torch.prelu(
            input_49,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_49 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_51 = torch.nn.functional.group_norm(
            input_50,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_50 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_52 = torch.conv1d(
            input_51,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (1,),
            (1,),
            512,
        )
        input_51 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_53 = torch.prelu(
            input_52,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_52 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_54 = torch.nn.functional.group_norm(
            input_53,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_53 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_8 = torch.conv1d(
            input_54,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_res_out_parameters_bias_ = (None)
        skip_out_8 = torch.conv1d(
            input_54,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_54 = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_8_modules_skip_out_parameters_bias_ = (None)
        feats_11 = feats_10 + residual_8
        feats_10 = residual_8 = None
        output_8 = output_7 + skip_out_8
        output_7 = skip_out_8 = None
        input_55 = torch.conv1d(
            feats_11,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_56 = torch.prelu(
            input_55,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_55 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_57 = torch.nn.functional.group_norm(
            input_56,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_56 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_58 = torch.conv1d(
            input_57,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (2,),
            (2,),
            512,
        )
        input_57 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_59 = torch.prelu(
            input_58,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_58 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_60 = torch.nn.functional.group_norm(
            input_59,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_59 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_9 = torch.conv1d(
            input_60,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_res_out_parameters_bias_ = (None)
        skip_out_9 = torch.conv1d(
            input_60,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_60 = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_9_modules_skip_out_parameters_bias_ = (None)
        feats_12 = feats_11 + residual_9
        feats_11 = residual_9 = None
        output_9 = output_8 + skip_out_9
        output_8 = skip_out_9 = None
        input_61 = torch.conv1d(
            feats_12,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_62 = torch.prelu(
            input_61,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_61 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_63 = torch.nn.functional.group_norm(
            input_62,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_62 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_64 = torch.conv1d(
            input_63,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (4,),
            (4,),
            512,
        )
        input_63 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_65 = torch.prelu(
            input_64,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_64 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_66 = torch.nn.functional.group_norm(
            input_65,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_65 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_10 = torch.conv1d(
            input_66,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_res_out_parameters_bias_ = (None)
        skip_out_10 = torch.conv1d(
            input_66,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_66 = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_10_modules_skip_out_parameters_bias_ = (None)
        feats_13 = feats_12 + residual_10
        feats_12 = residual_10 = None
        output_10 = output_9 + skip_out_10
        output_9 = skip_out_10 = None
        input_67 = torch.conv1d(
            feats_13,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_68 = torch.prelu(
            input_67,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_67 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_69 = torch.nn.functional.group_norm(
            input_68,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_68 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_70 = torch.conv1d(
            input_69,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (8,),
            (8,),
            512,
        )
        input_69 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_71 = torch.prelu(
            input_70,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_70 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_72 = torch.nn.functional.group_norm(
            input_71,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_71 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_11 = torch.conv1d(
            input_72,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_res_out_parameters_bias_ = (None)
        skip_out_11 = torch.conv1d(
            input_72,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_72 = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_11_modules_skip_out_parameters_bias_ = (None)
        feats_14 = feats_13 + residual_11
        feats_13 = residual_11 = None
        output_11 = output_10 + skip_out_11
        output_10 = skip_out_11 = None
        input_73 = torch.conv1d(
            feats_14,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_74 = torch.prelu(
            input_73,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_73 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_75 = torch.nn.functional.group_norm(
            input_74,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_74 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_76 = torch.conv1d(
            input_75,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (16,),
            (16,),
            512,
        )
        input_75 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_77 = torch.prelu(
            input_76,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_76 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_78 = torch.nn.functional.group_norm(
            input_77,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_77 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_12 = torch.conv1d(
            input_78,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_res_out_parameters_bias_ = (None)
        skip_out_12 = torch.conv1d(
            input_78,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_78 = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_12_modules_skip_out_parameters_bias_ = (None)
        feats_15 = feats_14 + residual_12
        feats_14 = residual_12 = None
        output_12 = output_11 + skip_out_12
        output_11 = skip_out_12 = None
        input_79 = torch.conv1d(
            feats_15,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_80 = torch.prelu(
            input_79,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_79 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_81 = torch.nn.functional.group_norm(
            input_80,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_80 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_82 = torch.conv1d(
            input_81,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (32,),
            (32,),
            512,
        )
        input_81 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_83 = torch.prelu(
            input_82,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_82 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_84 = torch.nn.functional.group_norm(
            input_83,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_83 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_13 = torch.conv1d(
            input_84,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_res_out_parameters_bias_ = (None)
        skip_out_13 = torch.conv1d(
            input_84,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_84 = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_13_modules_skip_out_parameters_bias_ = (None)
        feats_16 = feats_15 + residual_13
        feats_15 = residual_13 = None
        output_13 = output_12 + skip_out_13
        output_12 = skip_out_13 = None
        input_85 = torch.conv1d(
            feats_16,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_86 = torch.prelu(
            input_85,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_85 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_87 = torch.nn.functional.group_norm(
            input_86,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_86 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_88 = torch.conv1d(
            input_87,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (64,),
            (64,),
            512,
        )
        input_87 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_89 = torch.prelu(
            input_88,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_88 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_90 = torch.nn.functional.group_norm(
            input_89,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_89 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_14 = torch.conv1d(
            input_90,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_res_out_parameters_bias_ = (None)
        skip_out_14 = torch.conv1d(
            input_90,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_90 = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_14_modules_skip_out_parameters_bias_ = (None)
        feats_17 = feats_16 + residual_14
        feats_16 = residual_14 = None
        output_14 = output_13 + skip_out_14
        output_13 = skip_out_14 = None
        input_91 = torch.conv1d(
            feats_17,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_92 = torch.prelu(
            input_91,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_91 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_93 = torch.nn.functional.group_norm(
            input_92,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_92 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_94 = torch.conv1d(
            input_93,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (128,),
            (128,),
            512,
        )
        input_93 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_95 = torch.prelu(
            input_94,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_94 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_96 = torch.nn.functional.group_norm(
            input_95,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_95 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_15 = torch.conv1d(
            input_96,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_res_out_parameters_bias_ = (None)
        skip_out_15 = torch.conv1d(
            input_96,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_96 = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_15_modules_skip_out_parameters_bias_ = (None)
        feats_18 = feats_17 + residual_15
        feats_17 = residual_15 = None
        output_15 = output_14 + skip_out_15
        output_14 = skip_out_15 = None
        input_97 = torch.conv1d(
            feats_18,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_98 = torch.prelu(
            input_97,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_97 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_99 = torch.nn.functional.group_norm(
            input_98,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_98 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_100 = torch.conv1d(
            input_99,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (1,),
            (1,),
            512,
        )
        input_99 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_101 = torch.prelu(
            input_100,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_100 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_102 = torch.nn.functional.group_norm(
            input_101,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_101 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_16 = torch.conv1d(
            input_102,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_res_out_parameters_bias_ = (None)
        skip_out_16 = torch.conv1d(
            input_102,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_102 = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_16_modules_skip_out_parameters_bias_ = (None)
        feats_19 = feats_18 + residual_16
        feats_18 = residual_16 = None
        output_16 = output_15 + skip_out_16
        output_15 = skip_out_16 = None
        input_103 = torch.conv1d(
            feats_19,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_104 = torch.prelu(
            input_103,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_103 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_105 = torch.nn.functional.group_norm(
            input_104,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_104 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_106 = torch.conv1d(
            input_105,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (2,),
            (2,),
            512,
        )
        input_105 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_107 = torch.prelu(
            input_106,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_106 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_108 = torch.nn.functional.group_norm(
            input_107,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_107 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_17 = torch.conv1d(
            input_108,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_res_out_parameters_bias_ = (None)
        skip_out_17 = torch.conv1d(
            input_108,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_108 = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_17_modules_skip_out_parameters_bias_ = (None)
        feats_20 = feats_19 + residual_17
        feats_19 = residual_17 = None
        output_17 = output_16 + skip_out_17
        output_16 = skip_out_17 = None
        input_109 = torch.conv1d(
            feats_20,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_110 = torch.prelu(
            input_109,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_109 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_111 = torch.nn.functional.group_norm(
            input_110,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_110 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_112 = torch.conv1d(
            input_111,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (4,),
            (4,),
            512,
        )
        input_111 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_113 = torch.prelu(
            input_112,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_112 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_114 = torch.nn.functional.group_norm(
            input_113,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_113 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_18 = torch.conv1d(
            input_114,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_res_out_parameters_bias_ = (None)
        skip_out_18 = torch.conv1d(
            input_114,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_114 = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_18_modules_skip_out_parameters_bias_ = (None)
        feats_21 = feats_20 + residual_18
        feats_20 = residual_18 = None
        output_18 = output_17 + skip_out_18
        output_17 = skip_out_18 = None
        input_115 = torch.conv1d(
            feats_21,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_116 = torch.prelu(
            input_115,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_115 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_117 = torch.nn.functional.group_norm(
            input_116,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_116 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_118 = torch.conv1d(
            input_117,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (8,),
            (8,),
            512,
        )
        input_117 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_119 = torch.prelu(
            input_118,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_118 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_120 = torch.nn.functional.group_norm(
            input_119,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_119 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_19 = torch.conv1d(
            input_120,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_res_out_parameters_bias_ = (None)
        skip_out_19 = torch.conv1d(
            input_120,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_120 = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_19_modules_skip_out_parameters_bias_ = (None)
        feats_22 = feats_21 + residual_19
        feats_21 = residual_19 = None
        output_19 = output_18 + skip_out_19
        output_18 = skip_out_19 = None
        input_121 = torch.conv1d(
            feats_22,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_122 = torch.prelu(
            input_121,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_121 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_123 = torch.nn.functional.group_norm(
            input_122,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_122 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_124 = torch.conv1d(
            input_123,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (16,),
            (16,),
            512,
        )
        input_123 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_125 = torch.prelu(
            input_124,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_124 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_126 = torch.nn.functional.group_norm(
            input_125,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_125 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_20 = torch.conv1d(
            input_126,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_res_out_parameters_bias_ = (None)
        skip_out_20 = torch.conv1d(
            input_126,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_126 = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_20_modules_skip_out_parameters_bias_ = (None)
        feats_23 = feats_22 + residual_20
        feats_22 = residual_20 = None
        output_20 = output_19 + skip_out_20
        output_19 = skip_out_20 = None
        input_127 = torch.conv1d(
            feats_23,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_128 = torch.prelu(
            input_127,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_127 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_129 = torch.nn.functional.group_norm(
            input_128,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_128 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_130 = torch.conv1d(
            input_129,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (32,),
            (32,),
            512,
        )
        input_129 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_131 = torch.prelu(
            input_130,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_130 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_132 = torch.nn.functional.group_norm(
            input_131,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_131 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_21 = torch.conv1d(
            input_132,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_res_out_parameters_bias_ = (None)
        skip_out_21 = torch.conv1d(
            input_132,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_132 = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_21_modules_skip_out_parameters_bias_ = (None)
        feats_24 = feats_23 + residual_21
        feats_23 = residual_21 = None
        output_21 = output_20 + skip_out_21
        output_20 = skip_out_21 = None
        input_133 = torch.conv1d(
            feats_24,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_134 = torch.prelu(
            input_133,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_133 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_135 = torch.nn.functional.group_norm(
            input_134,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_134 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_136 = torch.conv1d(
            input_135,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (64,),
            (64,),
            512,
        )
        input_135 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_137 = torch.prelu(
            input_136,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_136 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_138 = torch.nn.functional.group_norm(
            input_137,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_137 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_conv_layers_modules_5_parameters_bias_ = (None)
        residual_22 = torch.conv1d(
            input_138,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_res_out_parameters_bias_ = (None)
        skip_out_22 = torch.conv1d(
            input_138,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_138 = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_22_modules_skip_out_parameters_bias_ = (None)
        feats_25 = feats_24 + residual_22
        feats_24 = residual_22 = None
        output_22 = output_21 + skip_out_22
        output_21 = skip_out_22 = None
        input_139 = torch.conv1d(
            feats_25,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        feats_25 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_0_parameters_bias_ = (None)
        input_140 = torch.prelu(
            input_139,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_1_parameters_weight_,
        )
        input_139 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_1_parameters_weight_ = (None)
        input_141 = torch.nn.functional.group_norm(
            input_140,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_bias_,
            1e-08,
        )
        input_140 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_2_parameters_bias_ = (None)
        input_142 = torch.conv1d(
            input_141,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_bias_,
            (1,),
            (128,),
            (128,),
            512,
        )
        input_141 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_3_parameters_bias_ = (None)
        input_143 = torch.prelu(
            input_142,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_4_parameters_weight_,
        )
        input_142 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_4_parameters_weight_ = (None)
        input_144 = torch.nn.functional.group_norm(
            input_143,
            1,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_bias_,
            1e-08,
        )
        input_143 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_conv_layers_modules_5_parameters_bias_ = (None)
        skip_out_23 = torch.conv1d(
            input_144,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_weight_,
            l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        input_144 = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_weight_ = l_self_modules_mask_generator_modules_conv_layers_modules_23_modules_skip_out_parameters_bias_ = (None)
        output_23 = output_22 + skip_out_23
        output_22 = skip_out_23 = None
        output_24 = torch.prelu(
            output_23,
            l_self_modules_mask_generator_modules_output_prelu_parameters_weight_,
        )
        output_23 = (
            l_self_modules_mask_generator_modules_output_prelu_parameters_weight_
        ) = None
        output_25 = torch.conv1d(
            output_24,
            l_self_modules_mask_generator_modules_output_conv_parameters_weight_,
            l_self_modules_mask_generator_modules_output_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        output_24 = (
            l_self_modules_mask_generator_modules_output_conv_parameters_weight_
        ) = l_self_modules_mask_generator_modules_output_conv_parameters_bias_ = None
        output_26 = torch.nn.functional.relu(output_25, inplace=False)
        output_25 = None
        view = output_26.view(1, 2, 512, -1)
        output_26 = None
        unsqueeze = feats.unsqueeze(1)
        feats = None
        masked = view * unsqueeze
        view = unsqueeze = None
        masked_1 = masked.view(2, 512, -1)
        masked = None
        decoded = torch.conv_transpose1d(
            masked_1,
            l_self_modules_decoder_parameters_weight_,
            None,
            (8,),
            (8,),
            (0,),
            1,
            (1,),
        )
        masked_1 = l_self_modules_decoder_parameters_weight_ = None
        output_27 = decoded.view(1, 2, 32000)
        decoded = None
        return (output_27,)
