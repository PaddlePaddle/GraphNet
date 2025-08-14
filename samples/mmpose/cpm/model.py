import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_4_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_6_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_6_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_7_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_7_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_8_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_8_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_bias_
        )
        l_self_modules_backbone_modules_middle_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_middle_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_middle_modules_4_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_4_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_bias_
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        input_1 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        x_3 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_2 = torch.nn.functional.max_pool2d(
            x_5, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_5 = None
        x_6 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_stem_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_backbone_modules_stem_modules_4_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_4_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        input_3 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_9 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_stem_modules_6_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_backbone_modules_stem_modules_6_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_6_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_6_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_12 = torch.conv2d(
            x_11,
            l_self_modules_backbone_modules_stem_modules_7_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_backbone_modules_stem_modules_7_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_7_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_7_modules_bn_parameters_bias_
        ) = None
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_backbone_modules_stem_modules_8_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_backbone_modules_stem_modules_8_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_8_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_8_modules_bn_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_weight_ = (
            l_self_modules_backbone_modules_stem_modules_9_modules_conv_parameters_bias_
        ) = None
        x_19 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_middle_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_middle_modules_0_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_middle_modules_0_modules_bn_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        input_4 = torch.nn.functional.max_pool2d(
            x_21, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_21 = None
        x_22 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_middle_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_backbone_modules_middle_modules_2_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_middle_modules_2_modules_bn_parameters_bias_
        ) = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        input_5 = torch.nn.functional.max_pool2d(
            x_24, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_24 = None
        x_25 = torch.conv2d(
            input_5,
            l_self_modules_backbone_modules_middle_modules_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_backbone_modules_middle_modules_4_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_modules_4_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_middle_modules_4_modules_bn_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        input_6 = torch.nn.functional.max_pool2d(
            x_27, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_27 = None
        x_28 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_middle_conv_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        inp_feat = torch.cat([x_18, x_30], 1)
        x_30 = None
        x_31 = torch.conv2d(
            inp_feat,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        inp_feat = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_0_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_1_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_0_modules_model_modules_2_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_0_modules_0_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_0_modules_1_modules_conv_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_middle_conv_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        inp_feat_1 = torch.cat([x_43, x_46], 1)
        x_46 = None
        x_47 = torch.conv2d(
            inp_feat_1,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        inp_feat_1 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_0_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_1_modules_model_modules_2_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_1_modules_0_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_1_modules_1_modules_conv_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_middle_conv_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        inp_feat_2 = torch.cat([x_59, x_62], 1)
        x_62 = None
        x_63 = torch.conv2d(
            inp_feat_2,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        inp_feat_2 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_0_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_1_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_2_modules_model_modules_2_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_2_modules_0_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_2_modules_1_modules_conv_parameters_bias_ = (None)
        x_76 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_middle_conv_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        inp_feat_3 = torch.cat([x_75, x_78], 1)
        x_78 = None
        x_79 = torch.conv2d(
            inp_feat_3,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        inp_feat_3 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_0_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_1_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_3_modules_model_modules_2_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_3_modules_0_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_3_modules_1_modules_conv_parameters_bias_ = (None)
        x_92 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_middle_conv_modules_4_modules_0_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        inp_feat_4 = torch.cat([x_91, x_94], 1)
        x_94 = None
        x_95 = torch.conv2d(
            inp_feat_4,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        inp_feat_4 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_conv_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_0_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_1_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (5, 5),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_conv_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_cpm_stages_modules_4_modules_model_modules_2_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_4_modules_0_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_out_convs_modules_4_modules_1_modules_conv_parameters_bias_ = (None)
        return (x_18, x_43, x_59, x_75, x_91, x_107)
