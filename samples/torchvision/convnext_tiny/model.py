import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_parameters_layer_scale_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_0_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_1_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_1_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_1_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_2_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_3_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_3_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_3_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_4_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_3_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_3_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_4_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_4_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_5_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_5_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_6_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_6_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_7_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_7_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_5_modules_8_parameters_layer_scale_ = (
            L_self_modules_features_modules_5_modules_8_parameters_layer_scale_
        )
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_6_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_0_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_0_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_0_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_0_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_7_modules_1_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_1_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_
        l_self_modules_features_modules_7_modules_2_parameters_layer_scale_ = (
            L_self_modules_features_modules_7_modules_2_parameters_layer_scale_
        )
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_
        l_self_modules_classifier_modules_0_parameters_weight_ = (
            L_self_modules_classifier_modules_0_parameters_weight_
        )
        l_self_modules_classifier_modules_0_parameters_bias_ = (
            L_self_modules_classifier_modules_0_parameters_bias_
        )
        l_self_modules_classifier_modules_2_parameters_weight_ = (
            L_self_modules_classifier_modules_2_parameters_weight_
        )
        l_self_modules_classifier_modules_2_parameters_bias_ = (
            L_self_modules_classifier_modules_2_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_modules_0_parameters_weight_,
            l_self_modules_features_modules_0_modules_0_parameters_bias_,
            (4, 4),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_features_modules_0_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_0_parameters_bias_ = None
        x = input_1.permute(0, 2, 3, 1)
        input_1 = None
        x_1 = torch.nn.functional.layer_norm(
            x,
            (96,),
            l_self_modules_features_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_1_parameters_bias_,
            1e-06,
        )
        x = (
            l_self_modules_features_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_1_parameters_bias_ = None
        x_2 = x_1.permute(0, 3, 1, 2)
        x_1 = None
        input_2 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_3 = torch.permute(input_2, [0, 2, 3, 1])
        input_2 = None
        input_4 = torch.nn.functional.layer_norm(
            input_3,
            (96,),
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_3 = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_5 = torch._C._nn.linear(
            input_4,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_4 = l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_6 = torch._C._nn.gelu(input_5, approximate="none")
        input_5 = None
        input_7 = torch._C._nn.linear(
            input_6,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_6 = l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_8 = torch.permute(input_7, [0, 3, 1, 2])
        input_7 = None
        result = (
            l_self_modules_features_modules_1_modules_0_parameters_layer_scale_
            * input_8
        )
        l_self_modules_features_modules_1_modules_0_parameters_layer_scale_ = (
            input_8
        ) = None
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        result += x_2
        result_1 = result
        result = x_2 = None
        input_9 = torch.conv2d(
            result_1,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_10 = torch.permute(input_9, [0, 2, 3, 1])
        input_9 = None
        input_11 = torch.nn.functional.layer_norm(
            input_10,
            (96,),
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_10 = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_12 = torch._C._nn.linear(
            input_11,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_11 = l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_13 = torch._C._nn.gelu(input_12, approximate="none")
        input_12 = None
        input_14 = torch._C._nn.linear(
            input_13,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_13 = l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_15 = torch.permute(input_14, [0, 3, 1, 2])
        input_14 = None
        result_2 = (
            l_self_modules_features_modules_1_modules_1_parameters_layer_scale_
            * input_15
        )
        l_self_modules_features_modules_1_modules_1_parameters_layer_scale_ = (
            input_15
        ) = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        result_2 += result_1
        result_3 = result_2
        result_2 = result_1 = None
        input_16 = torch.conv2d(
            result_3,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            96,
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_17 = torch.permute(input_16, [0, 2, 3, 1])
        input_16 = None
        input_18 = torch.nn.functional.layer_norm(
            input_17,
            (96,),
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_17 = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_18 = l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_20 = torch._C._nn.gelu(input_19, approximate="none")
        input_19 = None
        input_21 = torch._C._nn.linear(
            input_20,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_20 = l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_22 = torch.permute(input_21, [0, 3, 1, 2])
        input_21 = None
        result_4 = (
            l_self_modules_features_modules_1_modules_2_parameters_layer_scale_
            * input_22
        )
        l_self_modules_features_modules_1_modules_2_parameters_layer_scale_ = (
            input_22
        ) = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        result_4 += result_3
        result_5 = result_4
        result_4 = result_3 = None
        x_3 = result_5.permute(0, 2, 3, 1)
        result_5 = None
        x_4 = torch.nn.functional.layer_norm(
            x_3,
            (96,),
            l_self_modules_features_modules_2_modules_0_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_parameters_bias_,
            1e-06,
        )
        x_3 = (
            l_self_modules_features_modules_2_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_0_parameters_bias_ = None
        x_5 = x_4.permute(0, 3, 1, 2)
        x_4 = None
        input_23 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_features_modules_2_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_2_modules_1_parameters_bias_ = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_25 = torch.permute(input_24, [0, 2, 3, 1])
        input_24 = None
        input_26 = torch.nn.functional.layer_norm(
            input_25,
            (192,),
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_25 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_27 = torch._C._nn.linear(
            input_26,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_26 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_28 = torch._C._nn.gelu(input_27, approximate="none")
        input_27 = None
        input_29 = torch._C._nn.linear(
            input_28,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_28 = l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_30 = torch.permute(input_29, [0, 3, 1, 2])
        input_29 = None
        result_6 = (
            l_self_modules_features_modules_3_modules_0_parameters_layer_scale_
            * input_30
        )
        l_self_modules_features_modules_3_modules_0_parameters_layer_scale_ = (
            input_30
        ) = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        result_6 += input_23
        result_7 = result_6
        result_6 = input_23 = None
        input_31 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_32 = torch.permute(input_31, [0, 2, 3, 1])
        input_31 = None
        input_33 = torch.nn.functional.layer_norm(
            input_32,
            (192,),
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_32 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_34 = torch._C._nn.linear(
            input_33,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_33 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_35 = torch._C._nn.gelu(input_34, approximate="none")
        input_34 = None
        input_36 = torch._C._nn.linear(
            input_35,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_35 = l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_37 = torch.permute(input_36, [0, 3, 1, 2])
        input_36 = None
        result_8 = (
            l_self_modules_features_modules_3_modules_1_parameters_layer_scale_
            * input_37
        )
        l_self_modules_features_modules_3_modules_1_parameters_layer_scale_ = (
            input_37
        ) = None
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        result_8 += result_7
        result_9 = result_8
        result_8 = result_7 = None
        input_38 = torch.conv2d(
            result_9,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            192,
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_39 = torch.permute(input_38, [0, 2, 3, 1])
        input_38 = None
        input_40 = torch.nn.functional.layer_norm(
            input_39,
            (192,),
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_39 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_41 = torch._C._nn.linear(
            input_40,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_40 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_42 = torch._C._nn.gelu(input_41, approximate="none")
        input_41 = None
        input_43 = torch._C._nn.linear(
            input_42,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_42 = l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_44 = torch.permute(input_43, [0, 3, 1, 2])
        input_43 = None
        result_10 = (
            l_self_modules_features_modules_3_modules_2_parameters_layer_scale_
            * input_44
        )
        l_self_modules_features_modules_3_modules_2_parameters_layer_scale_ = (
            input_44
        ) = None
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        result_10 += result_9
        result_11 = result_10
        result_10 = result_9 = None
        x_6 = result_11.permute(0, 2, 3, 1)
        result_11 = None
        x_7 = torch.nn.functional.layer_norm(
            x_6,
            (192,),
            l_self_modules_features_modules_4_modules_0_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_parameters_bias_,
            1e-06,
        )
        x_6 = (
            l_self_modules_features_modules_4_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_0_parameters_bias_ = None
        x_8 = x_7.permute(0, 3, 1, 2)
        x_7 = None
        input_45 = torch.conv2d(
            x_8,
            l_self_modules_features_modules_4_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_features_modules_4_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_4_modules_1_parameters_bias_ = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_47 = torch.permute(input_46, [0, 2, 3, 1])
        input_46 = None
        input_48 = torch.nn.functional.layer_norm(
            input_47,
            (384,),
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_47 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_49 = torch._C._nn.linear(
            input_48,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_48 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_50 = torch._C._nn.gelu(input_49, approximate="none")
        input_49 = None
        input_51 = torch._C._nn.linear(
            input_50,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_50 = l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_52 = torch.permute(input_51, [0, 3, 1, 2])
        input_51 = None
        result_12 = (
            l_self_modules_features_modules_5_modules_0_parameters_layer_scale_
            * input_52
        )
        l_self_modules_features_modules_5_modules_0_parameters_layer_scale_ = (
            input_52
        ) = None
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        result_12 += input_45
        result_13 = result_12
        result_12 = input_45 = None
        input_53 = torch.conv2d(
            result_13,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_54 = torch.permute(input_53, [0, 2, 3, 1])
        input_53 = None
        input_55 = torch.nn.functional.layer_norm(
            input_54,
            (384,),
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_54 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_56 = torch._C._nn.linear(
            input_55,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_55 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_57 = torch._C._nn.gelu(input_56, approximate="none")
        input_56 = None
        input_58 = torch._C._nn.linear(
            input_57,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_57 = l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_59 = torch.permute(input_58, [0, 3, 1, 2])
        input_58 = None
        result_14 = (
            l_self_modules_features_modules_5_modules_1_parameters_layer_scale_
            * input_59
        )
        l_self_modules_features_modules_5_modules_1_parameters_layer_scale_ = (
            input_59
        ) = None
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        result_14 += result_13
        result_15 = result_14
        result_14 = result_13 = None
        input_60 = torch.conv2d(
            result_15,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_61 = torch.permute(input_60, [0, 2, 3, 1])
        input_60 = None
        input_62 = torch.nn.functional.layer_norm(
            input_61,
            (384,),
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_61 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_63 = torch._C._nn.linear(
            input_62,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_62 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_64 = torch._C._nn.gelu(input_63, approximate="none")
        input_63 = None
        input_65 = torch._C._nn.linear(
            input_64,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_64 = l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_66 = torch.permute(input_65, [0, 3, 1, 2])
        input_65 = None
        result_16 = (
            l_self_modules_features_modules_5_modules_2_parameters_layer_scale_
            * input_66
        )
        l_self_modules_features_modules_5_modules_2_parameters_layer_scale_ = (
            input_66
        ) = None
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        result_16 += result_15
        result_17 = result_16
        result_16 = result_15 = None
        input_67 = torch.conv2d(
            result_17,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_ = (None)
        input_68 = torch.permute(input_67, [0, 2, 3, 1])
        input_67 = None
        input_69 = torch.nn.functional.layer_norm(
            input_68,
            (384,),
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_68 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_ = (None)
        input_70 = torch._C._nn.linear(
            input_69,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_,
        )
        input_69 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_ = (None)
        input_71 = torch._C._nn.gelu(input_70, approximate="none")
        input_70 = None
        input_72 = torch._C._nn.linear(
            input_71,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_,
        )
        input_71 = l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_ = (None)
        input_73 = torch.permute(input_72, [0, 3, 1, 2])
        input_72 = None
        result_18 = (
            l_self_modules_features_modules_5_modules_3_parameters_layer_scale_
            * input_73
        )
        l_self_modules_features_modules_5_modules_3_parameters_layer_scale_ = (
            input_73
        ) = None
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        result_18 += result_17
        result_19 = result_18
        result_18 = result_17 = None
        input_74 = torch.conv2d(
            result_19,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_ = (None)
        input_75 = torch.permute(input_74, [0, 2, 3, 1])
        input_74 = None
        input_76 = torch.nn.functional.layer_norm(
            input_75,
            (384,),
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_75 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_ = (None)
        input_77 = torch._C._nn.linear(
            input_76,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_,
        )
        input_76 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_ = (None)
        input_78 = torch._C._nn.gelu(input_77, approximate="none")
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_,
        )
        input_78 = l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_ = (None)
        input_80 = torch.permute(input_79, [0, 3, 1, 2])
        input_79 = None
        result_20 = (
            l_self_modules_features_modules_5_modules_4_parameters_layer_scale_
            * input_80
        )
        l_self_modules_features_modules_5_modules_4_parameters_layer_scale_ = (
            input_80
        ) = None
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        result_20 += result_19
        result_21 = result_20
        result_20 = result_19 = None
        input_81 = torch.conv2d(
            result_21,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_ = (None)
        input_82 = torch.permute(input_81, [0, 2, 3, 1])
        input_81 = None
        input_83 = torch.nn.functional.layer_norm(
            input_82,
            (384,),
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_82 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_ = (None)
        input_84 = torch._C._nn.linear(
            input_83,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_,
        )
        input_83 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_ = (None)
        input_85 = torch._C._nn.gelu(input_84, approximate="none")
        input_84 = None
        input_86 = torch._C._nn.linear(
            input_85,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_,
        )
        input_85 = l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_ = (None)
        input_87 = torch.permute(input_86, [0, 3, 1, 2])
        input_86 = None
        result_22 = (
            l_self_modules_features_modules_5_modules_5_parameters_layer_scale_
            * input_87
        )
        l_self_modules_features_modules_5_modules_5_parameters_layer_scale_ = (
            input_87
        ) = None
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        result_22 += result_21
        result_23 = result_22
        result_22 = result_21 = None
        input_88 = torch.conv2d(
            result_23,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_ = (None)
        input_89 = torch.permute(input_88, [0, 2, 3, 1])
        input_88 = None
        input_90 = torch.nn.functional.layer_norm(
            input_89,
            (384,),
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_89 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_ = (None)
        input_91 = torch._C._nn.linear(
            input_90,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_,
        )
        input_90 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_ = (None)
        input_92 = torch._C._nn.gelu(input_91, approximate="none")
        input_91 = None
        input_93 = torch._C._nn.linear(
            input_92,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_,
        )
        input_92 = l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_ = (None)
        input_94 = torch.permute(input_93, [0, 3, 1, 2])
        input_93 = None
        result_24 = (
            l_self_modules_features_modules_5_modules_6_parameters_layer_scale_
            * input_94
        )
        l_self_modules_features_modules_5_modules_6_parameters_layer_scale_ = (
            input_94
        ) = None
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        result_24 += result_23
        result_25 = result_24
        result_24 = result_23 = None
        input_95 = torch.conv2d(
            result_25,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_ = (None)
        input_96 = torch.permute(input_95, [0, 2, 3, 1])
        input_95 = None
        input_97 = torch.nn.functional.layer_norm(
            input_96,
            (384,),
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_96 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_ = (None)
        input_98 = torch._C._nn.linear(
            input_97,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_,
        )
        input_97 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_ = (None)
        input_99 = torch._C._nn.gelu(input_98, approximate="none")
        input_98 = None
        input_100 = torch._C._nn.linear(
            input_99,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_,
        )
        input_99 = l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_ = (None)
        input_101 = torch.permute(input_100, [0, 3, 1, 2])
        input_100 = None
        result_26 = (
            l_self_modules_features_modules_5_modules_7_parameters_layer_scale_
            * input_101
        )
        l_self_modules_features_modules_5_modules_7_parameters_layer_scale_ = (
            input_101
        ) = None
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        result_26 += result_25
        result_27 = result_26
        result_26 = result_25 = None
        input_102 = torch.conv2d(
            result_27,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            384,
        )
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_ = (None)
        input_103 = torch.permute(input_102, [0, 2, 3, 1])
        input_102 = None
        input_104 = torch.nn.functional.layer_norm(
            input_103,
            (384,),
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_103 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_ = (None)
        input_105 = torch._C._nn.linear(
            input_104,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_,
        )
        input_104 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_ = (None)
        input_106 = torch._C._nn.gelu(input_105, approximate="none")
        input_105 = None
        input_107 = torch._C._nn.linear(
            input_106,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_,
        )
        input_106 = l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_ = (None)
        input_108 = torch.permute(input_107, [0, 3, 1, 2])
        input_107 = None
        result_28 = (
            l_self_modules_features_modules_5_modules_8_parameters_layer_scale_
            * input_108
        )
        l_self_modules_features_modules_5_modules_8_parameters_layer_scale_ = (
            input_108
        ) = None
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        result_28 += result_27
        result_29 = result_28
        result_28 = result_27 = None
        x_9 = result_29.permute(0, 2, 3, 1)
        result_29 = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (384,),
            l_self_modules_features_modules_6_modules_0_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_parameters_bias_,
            1e-06,
        )
        x_9 = (
            l_self_modules_features_modules_6_modules_0_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_0_parameters_bias_ = None
        x_11 = x_10.permute(0, 3, 1, 2)
        x_10 = None
        input_109 = torch.conv2d(
            x_11,
            l_self_modules_features_modules_6_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = (
            l_self_modules_features_modules_6_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_6_modules_1_parameters_bias_ = None
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_ = (None)
        input_111 = torch.permute(input_110, [0, 2, 3, 1])
        input_110 = None
        input_112 = torch.nn.functional.layer_norm(
            input_111,
            (768,),
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_111 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_ = (None)
        input_113 = torch._C._nn.linear(
            input_112,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_,
        )
        input_112 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_ = (None)
        input_114 = torch._C._nn.gelu(input_113, approximate="none")
        input_113 = None
        input_115 = torch._C._nn.linear(
            input_114,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_,
        )
        input_114 = l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_ = (None)
        input_116 = torch.permute(input_115, [0, 3, 1, 2])
        input_115 = None
        result_30 = (
            l_self_modules_features_modules_7_modules_0_parameters_layer_scale_
            * input_116
        )
        l_self_modules_features_modules_7_modules_0_parameters_layer_scale_ = (
            input_116
        ) = None
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        result_30 += input_109
        result_31 = result_30
        result_30 = input_109 = None
        input_117 = torch.conv2d(
            result_31,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_ = (None)
        input_118 = torch.permute(input_117, [0, 2, 3, 1])
        input_117 = None
        input_119 = torch.nn.functional.layer_norm(
            input_118,
            (768,),
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_118 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_ = (None)
        input_120 = torch._C._nn.linear(
            input_119,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_,
        )
        input_119 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_ = (None)
        input_121 = torch._C._nn.gelu(input_120, approximate="none")
        input_120 = None
        input_122 = torch._C._nn.linear(
            input_121,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_,
        )
        input_121 = l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_ = (None)
        input_123 = torch.permute(input_122, [0, 3, 1, 2])
        input_122 = None
        result_32 = (
            l_self_modules_features_modules_7_modules_1_parameters_layer_scale_
            * input_123
        )
        l_self_modules_features_modules_7_modules_1_parameters_layer_scale_ = (
            input_123
        ) = None
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        result_32 += result_31
        result_33 = result_32
        result_32 = result_31 = None
        input_124 = torch.conv2d(
            result_33,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            768,
        )
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_ = (None)
        input_125 = torch.permute(input_124, [0, 2, 3, 1])
        input_124 = None
        input_126 = torch.nn.functional.layer_norm(
            input_125,
            (768,),
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_,
            1e-06,
        )
        input_125 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_ = (None)
        input_127 = torch._C._nn.linear(
            input_126,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_,
        )
        input_126 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_ = (None)
        input_128 = torch._C._nn.gelu(input_127, approximate="none")
        input_127 = None
        input_129 = torch._C._nn.linear(
            input_128,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_,
        )
        input_128 = l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_ = (None)
        input_130 = torch.permute(input_129, [0, 3, 1, 2])
        input_129 = None
        result_34 = (
            l_self_modules_features_modules_7_modules_2_parameters_layer_scale_
            * input_130
        )
        l_self_modules_features_modules_7_modules_2_parameters_layer_scale_ = (
            input_130
        ) = None
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        result_34 += result_33
        result_35 = result_34
        result_34 = result_33 = None
        x_12 = torch.nn.functional.adaptive_avg_pool2d(result_35, 1)
        result_35 = None
        x_13 = x_12.permute(0, 2, 3, 1)
        x_12 = None
        x_14 = torch.nn.functional.layer_norm(
            x_13,
            (768,),
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
            1e-06,
        )
        x_13 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        x_15 = x_14.permute(0, 3, 1, 2)
        x_14 = None
        input_131 = x_15.flatten(1, -1)
        x_15 = None
        input_132 = torch._C._nn.linear(
            input_131,
            l_self_modules_classifier_modules_2_parameters_weight_,
            l_self_modules_classifier_modules_2_parameters_bias_,
        )
        input_131 = (
            l_self_modules_classifier_modules_2_parameters_weight_
        ) = l_self_modules_classifier_modules_2_parameters_bias_ = None
        return (input_132,)
