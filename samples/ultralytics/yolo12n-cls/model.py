import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_model_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_2_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_3_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_4_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_5_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_6_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_7_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_8_modules_cv2_modules_bn_eps: torch.Tensor,
        L_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_linear_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_9_modules_linear_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_model_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_0_modules_bn_momentum = (
            L_self_modules_model_modules_0_modules_bn_momentum
        )
        l_self_modules_model_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_0_modules_bn_eps = (
            L_self_modules_model_modules_0_modules_bn_eps
        )
        l_self_modules_model_modules_1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_1_modules_bn_momentum = (
            L_self_modules_model_modules_1_modules_bn_momentum
        )
        l_self_modules_model_modules_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_1_modules_bn_eps = (
            L_self_modules_model_modules_1_modules_bn_eps
        )
        l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_2_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_2_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_2_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_3_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_bn_momentum = (
            L_self_modules_model_modules_3_modules_bn_momentum
        )
        l_self_modules_model_modules_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_3_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_3_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_3_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_3_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_3_modules_bn_eps = (
            L_self_modules_model_modules_3_modules_bn_eps
        )
        l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_4_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps = L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps
        l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_4_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_4_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_5_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_bn_momentum = (
            L_self_modules_model_modules_5_modules_bn_momentum
        )
        l_self_modules_model_modules_5_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_5_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_5_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_5_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_5_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_5_modules_bn_eps = (
            L_self_modules_model_modules_5_modules_bn_eps
        )
        l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_6_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_6_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_6_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_7_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_bn_momentum = (
            L_self_modules_model_modules_7_modules_bn_momentum
        )
        l_self_modules_model_modules_7_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_7_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_7_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_7_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_7_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_7_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_7_modules_bn_eps = (
            L_self_modules_model_modules_7_modules_bn_eps
        )
        l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_cv1_modules_bn_eps = (
            L_self_modules_model_modules_8_modules_cv1_modules_bn_eps
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps = L_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps
        l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_8_modules_cv2_modules_bn_eps = (
            L_self_modules_model_modules_8_modules_cv2_modules_bn_eps
        )
        l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_model_modules_9_modules_linear_parameters_weight_ = (
            L_self_modules_model_modules_9_modules_linear_parameters_weight_
        )
        l_self_modules_model_modules_9_modules_linear_parameters_bias_ = (
            L_self_modules_model_modules_9_modules_linear_parameters_bias_
        )
        conv2d = torch.conv2d(
            l_x_,
            l_self_modules_model_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_model_modules_0_modules_conv_parameters_weight_ = None
        item = l_self_modules_model_modules_0_modules_bn_momentum.item()
        l_self_modules_model_modules_0_modules_bn_momentum = None
        item_1 = l_self_modules_model_modules_0_modules_bn_eps.item()
        l_self_modules_model_modules_0_modules_bn_eps = None
        batch_norm = torch.nn.functional.batch_norm(
            conv2d,
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_0_modules_bn_parameters_bias_,
            False,
            item,
            item_1,
        )
        conv2d = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_0_modules_bn_parameters_bias_
        ) = item = item_1 = None
        x = torch.nn.functional.silu(batch_norm, inplace=True)
        batch_norm = None
        conv2d_1 = torch.conv2d(
            x,
            l_self_modules_model_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x = l_self_modules_model_modules_1_modules_conv_parameters_weight_ = None
        item_2 = l_self_modules_model_modules_1_modules_bn_momentum.item()
        l_self_modules_model_modules_1_modules_bn_momentum = None
        item_3 = l_self_modules_model_modules_1_modules_bn_eps.item()
        l_self_modules_model_modules_1_modules_bn_eps = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_1_modules_bn_parameters_bias_,
            False,
            item_2,
            item_3,
        )
        conv2d_1 = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_1_modules_bn_parameters_bias_
        ) = item_2 = item_3 = None
        x_1 = torch.nn.functional.silu(batch_norm_1, inplace=True)
        batch_norm_1 = None
        conv2d_2 = torch.conv2d(
            x_1,
            l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_4 = l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_2_modules_cv1_modules_bn_momentum = None
        item_5 = l_self_modules_model_modules_2_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_2_modules_cv1_modules_bn_eps = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_4,
            item_5,
        )
        conv2d_2 = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_
        ) = item_4 = item_5 = None
        silu_2 = torch.nn.functional.silu(batch_norm_2, inplace=True)
        batch_norm_2 = None
        chunk = silu_2.chunk(2, 1)
        silu_2 = None
        getitem = chunk[0]
        getitem_1 = chunk[1]
        chunk = None
        conv2d_3 = torch.conv2d(
            getitem_1,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_6 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_7 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_6,
            item_7,
        )
        conv2d_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_6) = (
            item_7
        ) = None
        silu_3 = torch.nn.functional.silu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        conv2d_4 = torch.conv2d(
            silu_3,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_3 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_8 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_9 = (
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        conv2d_4 = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_8) = (
            item_9
        ) = None
        silu_4 = torch.nn.functional.silu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        add = getitem_1 + silu_4
        silu_4 = None
        cat = torch.cat([getitem, getitem_1, add], 1)
        getitem = getitem_1 = add = None
        conv2d_5 = torch.conv2d(
            cat,
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = (
            l_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_10 = l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_2_modules_cv2_modules_bn_momentum = None
        item_11 = l_self_modules_model_modules_2_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_2_modules_cv2_modules_bn_eps = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_10,
            item_11,
        )
        conv2d_5 = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_
        ) = item_10 = item_11 = None
        x_2 = torch.nn.functional.silu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        conv2d_6 = torch.conv2d(
            x_2,
            l_self_modules_model_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_model_modules_3_modules_conv_parameters_weight_ = None
        item_12 = l_self_modules_model_modules_3_modules_bn_momentum.item()
        l_self_modules_model_modules_3_modules_bn_momentum = None
        item_13 = l_self_modules_model_modules_3_modules_bn_eps.item()
        l_self_modules_model_modules_3_modules_bn_eps = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_3_modules_bn_parameters_weight_,
            l_self_modules_model_modules_3_modules_bn_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        conv2d_6 = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_3_modules_bn_parameters_bias_
        ) = item_12 = item_13 = None
        x_3 = torch.nn.functional.silu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_7 = torch.conv2d(
            x_3,
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = (
            l_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_14 = l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_4_modules_cv1_modules_bn_momentum = None
        item_15 = l_self_modules_model_modules_4_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_4_modules_cv1_modules_bn_eps = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        conv2d_7 = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_
        ) = item_14 = item_15 = None
        silu_7 = torch.nn.functional.silu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        chunk_1 = silu_7.chunk(2, 1)
        silu_7 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1]
        chunk_1 = None
        conv2d_8 = torch.conv2d(
            getitem_3,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_ = (
            None
        )
        item_16 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum = (
            None
        )
        item_17 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps = (
            None
        )
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_16,
            item_17,
        )
        conv2d_8 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_ = (item_16) = (
            item_17
        ) = None
        silu_8 = torch.nn.functional.silu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        conv2d_9 = torch.conv2d(
            silu_8,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        silu_8 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_ = (None)
        item_18 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum = (
            None
        )
        item_19 = (
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps.item()
        )
        l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps = (
            None
        )
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_18,
            item_19,
        )
        conv2d_9 = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_ = l_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_ = (item_18) = (
            item_19
        ) = None
        silu_9 = torch.nn.functional.silu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        add_1 = getitem_3 + silu_9
        silu_9 = None
        cat_1 = torch.cat([getitem_2, getitem_3, add_1], 1)
        getitem_2 = getitem_3 = add_1 = None
        conv2d_10 = torch.conv2d(
            cat_1,
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = (
            l_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_20 = l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_4_modules_cv2_modules_bn_momentum = None
        item_21 = l_self_modules_model_modules_4_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_4_modules_cv2_modules_bn_eps = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_20,
            item_21,
        )
        conv2d_10 = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_
        ) = item_20 = item_21 = None
        x_4 = torch.nn.functional.silu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        conv2d_11 = torch.conv2d(
            x_4,
            l_self_modules_model_modules_5_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_model_modules_5_modules_conv_parameters_weight_ = None
        item_22 = l_self_modules_model_modules_5_modules_bn_momentum.item()
        l_self_modules_model_modules_5_modules_bn_momentum = None
        item_23 = l_self_modules_model_modules_5_modules_bn_eps.item()
        l_self_modules_model_modules_5_modules_bn_eps = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_5_modules_bn_parameters_weight_,
            l_self_modules_model_modules_5_modules_bn_parameters_bias_,
            False,
            item_22,
            item_23,
        )
        conv2d_11 = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_5_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_5_modules_bn_parameters_bias_
        ) = item_22 = item_23 = None
        x_5 = torch.nn.functional.silu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        conv2d_12 = torch.conv2d(
            x_5,
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_24 = l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_6_modules_cv1_modules_bn_momentum = None
        item_25 = l_self_modules_model_modules_6_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_6_modules_cv1_modules_bn_eps = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        conv2d_12 = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_
        ) = item_24 = item_25 = None
        silu_12 = torch.nn.functional.silu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        conv2d_13 = torch.conv2d(
            silu_12,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_26 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_27 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_26,
            item_27,
        )
        conv2d_13 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_26) = (
            item_27
        ) = None
        flatten = batch_norm_13.flatten(2)
        batch_norm_13 = None
        qkv = flatten.transpose(1, 2)
        flatten = None
        qkv_1 = qkv.reshape(4, 400, 192)
        qkv = None
        view = qkv_1.view(4, 400, 2, 96)
        qkv_1 = None
        permute = view.permute(0, 2, 3, 1)
        view = None
        split = permute.split([32, 32, 32], dim=2)
        permute = None
        q = split[0]
        k = split[1]
        v = split[2]
        split = None
        transpose_1 = q.transpose(-2, -1)
        q = None
        matmul = transpose_1 @ k
        transpose_1 = k = None
        attn = matmul * 0.1767766952966369
        matmul = None
        attn_1 = attn.softmax(dim=-1)
        attn = None
        transpose_2 = attn_1.transpose(-2, -1)
        attn_1 = None
        x_6 = v @ transpose_2
        transpose_2 = None
        x_7 = x_6.permute(0, 3, 1, 2)
        x_6 = None
        v_1 = v.permute(0, 3, 1, 2)
        v = None
        x_8 = x_7.reshape(1, 1600, 64)
        x_7 = None
        v_2 = v_1.reshape(1, 1600, 64)
        v_1 = None
        reshape_3 = x_8.reshape(1, 40, 40, 64)
        x_8 = None
        permute_3 = reshape_3.permute(0, 3, 1, 2)
        reshape_3 = None
        x_9 = permute_3.contiguous()
        permute_3 = None
        reshape_4 = v_2.reshape(1, 40, 40, 64)
        v_2 = None
        permute_4 = reshape_4.permute(0, 3, 1, 2)
        reshape_4 = None
        v_3 = permute_4.contiguous()
        permute_4 = None
        conv2d_14 = torch.conv2d(
            v_3,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        v_3 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_28 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_29 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_28,
            item_29,
        )
        conv2d_14 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_28) = (
            item_29
        ) = None
        x_10 = x_9 + batch_norm_14
        x_9 = batch_norm_14 = None
        conv2d_15 = torch.conv2d(
            x_10,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_30 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_31 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_30,
            item_31,
        )
        conv2d_15 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_30) = (
            item_31
        ) = None
        x_11 = silu_12 + batch_norm_15
        batch_norm_15 = None
        conv2d_16 = torch.conv2d(
            x_11,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_32 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_33 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_32,
            item_33,
        )
        conv2d_16 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_32) = (
            item_33
        ) = None
        input_1 = torch.nn.functional.silu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        conv2d_17 = torch.conv2d(
            input_1,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_34 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_35 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_2 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_34,
            item_35,
        )
        conv2d_17 = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_34) = (
            item_35
        ) = None
        input_3 = x_11 + input_2
        x_11 = input_2 = None
        conv2d_18 = torch.conv2d(
            input_3,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_36 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_37 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_36,
            item_37,
        )
        conv2d_18 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_36) = (
            item_37
        ) = None
        flatten_1 = batch_norm_18.flatten(2)
        batch_norm_18 = None
        qkv_2 = flatten_1.transpose(1, 2)
        flatten_1 = None
        qkv_3 = qkv_2.reshape(4, 400, 192)
        qkv_2 = None
        view_1 = qkv_3.view(4, 400, 2, 96)
        qkv_3 = None
        permute_5 = view_1.permute(0, 2, 3, 1)
        view_1 = None
        split_1 = permute_5.split([32, 32, 32], dim=2)
        permute_5 = None
        q_1 = split_1[0]
        k_1 = split_1[1]
        v_4 = split_1[2]
        split_1 = None
        transpose_4 = q_1.transpose(-2, -1)
        q_1 = None
        matmul_2 = transpose_4 @ k_1
        transpose_4 = k_1 = None
        attn_2 = matmul_2 * 0.1767766952966369
        matmul_2 = None
        attn_3 = attn_2.softmax(dim=-1)
        attn_2 = None
        transpose_5 = attn_3.transpose(-2, -1)
        attn_3 = None
        x_12 = v_4 @ transpose_5
        transpose_5 = None
        x_13 = x_12.permute(0, 3, 1, 2)
        x_12 = None
        v_5 = v_4.permute(0, 3, 1, 2)
        v_4 = None
        x_14 = x_13.reshape(1, 1600, 64)
        x_13 = None
        v_6 = v_5.reshape(1, 1600, 64)
        v_5 = None
        reshape_8 = x_14.reshape(1, 40, 40, 64)
        x_14 = None
        permute_8 = reshape_8.permute(0, 3, 1, 2)
        reshape_8 = None
        x_15 = permute_8.contiguous()
        permute_8 = None
        reshape_9 = v_6.reshape(1, 40, 40, 64)
        v_6 = None
        permute_9 = reshape_9.permute(0, 3, 1, 2)
        reshape_9 = None
        v_7 = permute_9.contiguous()
        permute_9 = None
        conv2d_19 = torch.conv2d(
            v_7,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        v_7 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_38 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_39 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_38,
            item_39,
        )
        conv2d_19 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_38) = (
            item_39
        ) = None
        x_16 = x_15 + batch_norm_19
        x_15 = batch_norm_19 = None
        conv2d_20 = torch.conv2d(
            x_16,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_40 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_41 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_40,
            item_41,
        )
        conv2d_20 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_40) = (
            item_41
        ) = None
        x_17 = input_3 + batch_norm_20
        input_3 = batch_norm_20 = None
        conv2d_21 = torch.conv2d(
            x_17,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_42 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_43 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_42,
            item_43,
        )
        conv2d_21 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_42) = (
            item_43
        ) = None
        input_4 = torch.nn.functional.silu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        conv2d_22 = torch.conv2d(
            input_4,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_44 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_45 = (
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_44,
            item_45,
        )
        conv2d_22 = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_44) = (
            item_45
        ) = None
        input_6 = x_17 + input_5
        x_17 = input_5 = None
        conv2d_23 = torch.conv2d(
            silu_12,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_46 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_47 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_46,
            item_47,
        )
        conv2d_23 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_46) = (
            item_47
        ) = None
        flatten_2 = batch_norm_23.flatten(2)
        batch_norm_23 = None
        qkv_4 = flatten_2.transpose(1, 2)
        flatten_2 = None
        qkv_5 = qkv_4.reshape(4, 400, 192)
        qkv_4 = None
        view_2 = qkv_5.view(4, 400, 2, 96)
        qkv_5 = None
        permute_10 = view_2.permute(0, 2, 3, 1)
        view_2 = None
        split_2 = permute_10.split([32, 32, 32], dim=2)
        permute_10 = None
        q_2 = split_2[0]
        k_2 = split_2[1]
        v_8 = split_2[2]
        split_2 = None
        transpose_7 = q_2.transpose(-2, -1)
        q_2 = None
        matmul_4 = transpose_7 @ k_2
        transpose_7 = k_2 = None
        attn_4 = matmul_4 * 0.1767766952966369
        matmul_4 = None
        attn_5 = attn_4.softmax(dim=-1)
        attn_4 = None
        transpose_8 = attn_5.transpose(-2, -1)
        attn_5 = None
        x_18 = v_8 @ transpose_8
        transpose_8 = None
        x_19 = x_18.permute(0, 3, 1, 2)
        x_18 = None
        v_9 = v_8.permute(0, 3, 1, 2)
        v_8 = None
        x_20 = x_19.reshape(1, 1600, 64)
        x_19 = None
        v_10 = v_9.reshape(1, 1600, 64)
        v_9 = None
        reshape_13 = x_20.reshape(1, 40, 40, 64)
        x_20 = None
        permute_13 = reshape_13.permute(0, 3, 1, 2)
        reshape_13 = None
        x_21 = permute_13.contiguous()
        permute_13 = None
        reshape_14 = v_10.reshape(1, 40, 40, 64)
        v_10 = None
        permute_14 = reshape_14.permute(0, 3, 1, 2)
        reshape_14 = None
        v_11 = permute_14.contiguous()
        permute_14 = None
        conv2d_24 = torch.conv2d(
            v_11,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        v_11 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_48 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_49 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_48,
            item_49,
        )
        conv2d_24 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_48) = (
            item_49
        ) = None
        x_22 = x_21 + batch_norm_24
        x_21 = batch_norm_24 = None
        conv2d_25 = torch.conv2d(
            x_22,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_50 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_51 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_50,
            item_51,
        )
        conv2d_25 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_50) = (
            item_51
        ) = None
        x_23 = silu_12 + batch_norm_25
        batch_norm_25 = None
        conv2d_26 = torch.conv2d(
            x_23,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_52 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_53 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_52,
            item_53,
        )
        conv2d_26 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_52) = (
            item_53
        ) = None
        input_7 = torch.nn.functional.silu(batch_norm_26, inplace=True)
        batch_norm_26 = None
        conv2d_27 = torch.conv2d(
            input_7,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_54 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_55 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_8 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_54,
            item_55,
        )
        conv2d_27 = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_54) = (
            item_55
        ) = None
        input_9 = x_23 + input_8
        x_23 = input_8 = None
        conv2d_28 = torch.conv2d(
            input_9,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_56 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_57 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_56,
            item_57,
        )
        conv2d_28 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_56) = (
            item_57
        ) = None
        flatten_3 = batch_norm_28.flatten(2)
        batch_norm_28 = None
        qkv_6 = flatten_3.transpose(1, 2)
        flatten_3 = None
        qkv_7 = qkv_6.reshape(4, 400, 192)
        qkv_6 = None
        view_3 = qkv_7.view(4, 400, 2, 96)
        qkv_7 = None
        permute_15 = view_3.permute(0, 2, 3, 1)
        view_3 = None
        split_3 = permute_15.split([32, 32, 32], dim=2)
        permute_15 = None
        q_3 = split_3[0]
        k_3 = split_3[1]
        v_12 = split_3[2]
        split_3 = None
        transpose_10 = q_3.transpose(-2, -1)
        q_3 = None
        matmul_6 = transpose_10 @ k_3
        transpose_10 = k_3 = None
        attn_6 = matmul_6 * 0.1767766952966369
        matmul_6 = None
        attn_7 = attn_6.softmax(dim=-1)
        attn_6 = None
        transpose_11 = attn_7.transpose(-2, -1)
        attn_7 = None
        x_24 = v_12 @ transpose_11
        transpose_11 = None
        x_25 = x_24.permute(0, 3, 1, 2)
        x_24 = None
        v_13 = v_12.permute(0, 3, 1, 2)
        v_12 = None
        x_26 = x_25.reshape(1, 1600, 64)
        x_25 = None
        v_14 = v_13.reshape(1, 1600, 64)
        v_13 = None
        reshape_18 = x_26.reshape(1, 40, 40, 64)
        x_26 = None
        permute_18 = reshape_18.permute(0, 3, 1, 2)
        reshape_18 = None
        x_27 = permute_18.contiguous()
        permute_18 = None
        reshape_19 = v_14.reshape(1, 40, 40, 64)
        v_14 = None
        permute_19 = reshape_19.permute(0, 3, 1, 2)
        reshape_19 = None
        v_15 = permute_19.contiguous()
        permute_19 = None
        conv2d_29 = torch.conv2d(
            v_15,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        v_15 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_58 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_59 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_58,
            item_59,
        )
        conv2d_29 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_58) = (
            item_59
        ) = None
        x_28 = x_27 + batch_norm_29
        x_27 = batch_norm_29 = None
        conv2d_30 = torch.conv2d(
            x_28,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_60 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_61 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_60,
            item_61,
        )
        conv2d_30 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_60) = (
            item_61
        ) = None
        x_29 = input_9 + batch_norm_30
        input_9 = batch_norm_30 = None
        conv2d_31 = torch.conv2d(
            x_29,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_62 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_63 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_62,
            item_63,
        )
        conv2d_31 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_62) = (
            item_63
        ) = None
        input_10 = torch.nn.functional.silu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        conv2d_32 = torch.conv2d(
            input_10,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_64 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_65 = (
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_11 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_64,
            item_65,
        )
        conv2d_32 = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_6_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_64) = (
            item_65
        ) = None
        input_12 = x_29 + input_11
        x_29 = input_11 = None
        cat_2 = torch.cat([silu_12, input_6, input_12], 1)
        silu_12 = input_6 = input_12 = None
        conv2d_33 = torch.conv2d(
            cat_2,
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = (
            l_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_66 = l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_6_modules_cv2_modules_bn_momentum = None
        item_67 = l_self_modules_model_modules_6_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_6_modules_cv2_modules_bn_eps = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_66,
            item_67,
        )
        conv2d_33 = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_
        ) = item_66 = item_67 = None
        x_30 = torch.nn.functional.silu(batch_norm_33, inplace=True)
        batch_norm_33 = None
        conv2d_34 = torch.conv2d(
            x_30,
            l_self_modules_model_modules_7_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_model_modules_7_modules_conv_parameters_weight_ = None
        item_68 = l_self_modules_model_modules_7_modules_bn_momentum.item()
        l_self_modules_model_modules_7_modules_bn_momentum = None
        item_69 = l_self_modules_model_modules_7_modules_bn_eps.item()
        l_self_modules_model_modules_7_modules_bn_eps = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_7_modules_bn_parameters_weight_,
            l_self_modules_model_modules_7_modules_bn_parameters_bias_,
            False,
            item_68,
            item_69,
        )
        conv2d_34 = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_7_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_7_modules_bn_parameters_bias_
        ) = item_68 = item_69 = None
        x_31 = torch.nn.functional.silu(batch_norm_34, inplace=True)
        batch_norm_34 = None
        conv2d_35 = torch.conv2d(
            x_31,
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = (
            l_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_
        ) = None
        item_70 = l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum.item()
        l_self_modules_model_modules_8_modules_cv1_modules_bn_momentum = None
        item_71 = l_self_modules_model_modules_8_modules_cv1_modules_bn_eps.item()
        l_self_modules_model_modules_8_modules_cv1_modules_bn_eps = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_,
            False,
            item_70,
            item_71,
        )
        conv2d_35 = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_
        ) = item_70 = item_71 = None
        silu_19 = torch.nn.functional.silu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        conv2d_36 = torch.conv2d(
            silu_19,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_72 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_73 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_72,
            item_73,
        )
        conv2d_36 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_72) = (
            item_73
        ) = None
        flatten_4 = batch_norm_36.flatten(2)
        batch_norm_36 = None
        qkv_8 = flatten_4.transpose(1, 2)
        flatten_4 = None
        view_4 = qkv_8.view(1, 400, 4, 96)
        qkv_8 = None
        permute_20 = view_4.permute(0, 2, 3, 1)
        view_4 = None
        split_4 = permute_20.split([32, 32, 32], dim=2)
        permute_20 = None
        q_4 = split_4[0]
        k_4 = split_4[1]
        v_16 = split_4[2]
        split_4 = None
        transpose_13 = q_4.transpose(-2, -1)
        q_4 = None
        matmul_8 = transpose_13 @ k_4
        transpose_13 = k_4 = None
        attn_8 = matmul_8 * 0.1767766952966369
        matmul_8 = None
        attn_9 = attn_8.softmax(dim=-1)
        attn_8 = None
        transpose_14 = attn_9.transpose(-2, -1)
        attn_9 = None
        x_32 = v_16 @ transpose_14
        transpose_14 = None
        x_33 = x_32.permute(0, 3, 1, 2)
        x_32 = None
        v_17 = v_16.permute(0, 3, 1, 2)
        v_16 = None
        reshape_20 = x_33.reshape(1, 20, 20, 128)
        x_33 = None
        permute_23 = reshape_20.permute(0, 3, 1, 2)
        reshape_20 = None
        x_34 = permute_23.contiguous()
        permute_23 = None
        reshape_21 = v_17.reshape(1, 20, 20, 128)
        v_17 = None
        permute_24 = reshape_21.permute(0, 3, 1, 2)
        reshape_21 = None
        v_18 = permute_24.contiguous()
        permute_24 = None
        conv2d_37 = torch.conv2d(
            v_18,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        v_18 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_74 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_75 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_74,
            item_75,
        )
        conv2d_37 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_74) = (
            item_75
        ) = None
        x_35 = x_34 + batch_norm_37
        x_34 = batch_norm_37 = None
        conv2d_38 = torch.conv2d(
            x_35,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_76 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_77 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_76,
            item_77,
        )
        conv2d_38 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_76) = (
            item_77
        ) = None
        x_36 = silu_19 + batch_norm_38
        batch_norm_38 = None
        conv2d_39 = torch.conv2d(
            x_36,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_78 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_79 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_78,
            item_79,
        )
        conv2d_39 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_78) = (
            item_79
        ) = None
        input_13 = torch.nn.functional.silu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        conv2d_40 = torch.conv2d(
            input_13,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_80 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_81 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_14 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_80,
            item_81,
        )
        conv2d_40 = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_80) = (
            item_81
        ) = None
        input_15 = x_36 + input_14
        x_36 = input_14 = None
        conv2d_41 = torch.conv2d(
            input_15,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_82 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_83 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_82,
            item_83,
        )
        conv2d_41 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_82) = (
            item_83
        ) = None
        flatten_5 = batch_norm_41.flatten(2)
        batch_norm_41 = None
        qkv_9 = flatten_5.transpose(1, 2)
        flatten_5 = None
        view_5 = qkv_9.view(1, 400, 4, 96)
        qkv_9 = None
        permute_25 = view_5.permute(0, 2, 3, 1)
        view_5 = None
        split_5 = permute_25.split([32, 32, 32], dim=2)
        permute_25 = None
        q_5 = split_5[0]
        k_5 = split_5[1]
        v_19 = split_5[2]
        split_5 = None
        transpose_16 = q_5.transpose(-2, -1)
        q_5 = None
        matmul_10 = transpose_16 @ k_5
        transpose_16 = k_5 = None
        attn_10 = matmul_10 * 0.1767766952966369
        matmul_10 = None
        attn_11 = attn_10.softmax(dim=-1)
        attn_10 = None
        transpose_17 = attn_11.transpose(-2, -1)
        attn_11 = None
        x_37 = v_19 @ transpose_17
        transpose_17 = None
        x_38 = x_37.permute(0, 3, 1, 2)
        x_37 = None
        v_20 = v_19.permute(0, 3, 1, 2)
        v_19 = None
        reshape_22 = x_38.reshape(1, 20, 20, 128)
        x_38 = None
        permute_28 = reshape_22.permute(0, 3, 1, 2)
        reshape_22 = None
        x_39 = permute_28.contiguous()
        permute_28 = None
        reshape_23 = v_20.reshape(1, 20, 20, 128)
        v_20 = None
        permute_29 = reshape_23.permute(0, 3, 1, 2)
        reshape_23 = None
        v_21 = permute_29.contiguous()
        permute_29 = None
        conv2d_42 = torch.conv2d(
            v_21,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        v_21 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_84 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_85 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_84,
            item_85,
        )
        conv2d_42 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_84) = (
            item_85
        ) = None
        x_40 = x_39 + batch_norm_42
        x_39 = batch_norm_42 = None
        conv2d_43 = torch.conv2d(
            x_40,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_86 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_87 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_86,
            item_87,
        )
        conv2d_43 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_86) = (
            item_87
        ) = None
        x_41 = input_15 + batch_norm_43
        input_15 = batch_norm_43 = None
        conv2d_44 = torch.conv2d(
            x_41,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_88 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_89 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_88,
            item_89,
        )
        conv2d_44 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_88) = (
            item_89
        ) = None
        input_16 = torch.nn.functional.silu(batch_norm_44, inplace=True)
        batch_norm_44 = None
        conv2d_45 = torch.conv2d(
            input_16,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_90 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_91 = (
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_17 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_90,
            item_91,
        )
        conv2d_45 = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_0_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_90) = (
            item_91
        ) = None
        input_18 = x_41 + input_17
        x_41 = input_17 = None
        conv2d_46 = torch.conv2d(
            silu_19,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_92 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_93 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_46 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_92,
            item_93,
        )
        conv2d_46 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_92) = (
            item_93
        ) = None
        flatten_6 = batch_norm_46.flatten(2)
        batch_norm_46 = None
        qkv_10 = flatten_6.transpose(1, 2)
        flatten_6 = None
        view_6 = qkv_10.view(1, 400, 4, 96)
        qkv_10 = None
        permute_30 = view_6.permute(0, 2, 3, 1)
        view_6 = None
        split_6 = permute_30.split([32, 32, 32], dim=2)
        permute_30 = None
        q_6 = split_6[0]
        k_6 = split_6[1]
        v_22 = split_6[2]
        split_6 = None
        transpose_19 = q_6.transpose(-2, -1)
        q_6 = None
        matmul_12 = transpose_19 @ k_6
        transpose_19 = k_6 = None
        attn_12 = matmul_12 * 0.1767766952966369
        matmul_12 = None
        attn_13 = attn_12.softmax(dim=-1)
        attn_12 = None
        transpose_20 = attn_13.transpose(-2, -1)
        attn_13 = None
        x_42 = v_22 @ transpose_20
        transpose_20 = None
        x_43 = x_42.permute(0, 3, 1, 2)
        x_42 = None
        v_23 = v_22.permute(0, 3, 1, 2)
        v_22 = None
        reshape_24 = x_43.reshape(1, 20, 20, 128)
        x_43 = None
        permute_33 = reshape_24.permute(0, 3, 1, 2)
        reshape_24 = None
        x_44 = permute_33.contiguous()
        permute_33 = None
        reshape_25 = v_23.reshape(1, 20, 20, 128)
        v_23 = None
        permute_34 = reshape_25.permute(0, 3, 1, 2)
        reshape_25 = None
        v_24 = permute_34.contiguous()
        permute_34 = None
        conv2d_47 = torch.conv2d(
            v_24,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        v_24 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_94 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_95 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_94,
            item_95,
        )
        conv2d_47 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_94) = (
            item_95
        ) = None
        x_45 = x_44 + batch_norm_47
        x_44 = batch_norm_47 = None
        conv2d_48 = torch.conv2d(
            x_45,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_96 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_97 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_96,
            item_97,
        )
        conv2d_48 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_96) = (
            item_97
        ) = None
        x_46 = silu_19 + batch_norm_48
        batch_norm_48 = None
        conv2d_49 = torch.conv2d(
            x_46,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_98 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_99 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_98,
            item_99,
        )
        conv2d_49 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_98) = (
            item_99
        ) = None
        input_19 = torch.nn.functional.silu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        conv2d_50 = torch.conv2d(
            input_19,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_100 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_101 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_20 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_100,
            item_101,
        )
        conv2d_50 = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_0_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_100) = (
            item_101
        ) = None
        input_21 = x_46 + input_20
        x_46 = input_20 = None
        conv2d_51 = torch.conv2d(
            input_21,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_conv_parameters_weight_ = (
            None
        )
        item_102 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_momentum = (
            None
        )
        item_103 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_eps = (
            None
        )
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_,
            False,
            item_102,
            item_103,
        )
        conv2d_51 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_qkv_modules_bn_parameters_bias_ = (item_102) = (
            item_103
        ) = None
        flatten_7 = batch_norm_51.flatten(2)
        batch_norm_51 = None
        qkv_11 = flatten_7.transpose(1, 2)
        flatten_7 = None
        view_7 = qkv_11.view(1, 400, 4, 96)
        qkv_11 = None
        permute_35 = view_7.permute(0, 2, 3, 1)
        view_7 = None
        split_7 = permute_35.split([32, 32, 32], dim=2)
        permute_35 = None
        q_7 = split_7[0]
        k_7 = split_7[1]
        v_25 = split_7[2]
        split_7 = None
        transpose_22 = q_7.transpose(-2, -1)
        q_7 = None
        matmul_14 = transpose_22 @ k_7
        transpose_22 = k_7 = None
        attn_14 = matmul_14 * 0.1767766952966369
        matmul_14 = None
        attn_15 = attn_14.softmax(dim=-1)
        attn_14 = None
        transpose_23 = attn_15.transpose(-2, -1)
        attn_15 = None
        x_47 = v_25 @ transpose_23
        transpose_23 = None
        x_48 = x_47.permute(0, 3, 1, 2)
        x_47 = None
        v_26 = v_25.permute(0, 3, 1, 2)
        v_25 = None
        reshape_26 = x_48.reshape(1, 20, 20, 128)
        x_48 = None
        permute_38 = reshape_26.permute(0, 3, 1, 2)
        reshape_26 = None
        x_49 = permute_38.contiguous()
        permute_38 = None
        reshape_27 = v_26.reshape(1, 20, 20, 128)
        v_26 = None
        permute_39 = reshape_27.permute(0, 3, 1, 2)
        reshape_27 = None
        v_27 = permute_39.contiguous()
        permute_39 = None
        conv2d_52 = torch.conv2d(
            v_27,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        v_27 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_conv_parameters_weight_ = (None)
        item_104 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_momentum = (
            None
        )
        item_105 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_eps = (
            None
        )
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_,
            False,
            item_104,
            item_105,
        )
        conv2d_52 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_pe_modules_bn_parameters_bias_ = (item_104) = (
            item_105
        ) = None
        x_50 = x_49 + batch_norm_52
        x_49 = batch_norm_52 = None
        conv2d_53 = torch.conv2d(
            x_50,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_conv_parameters_weight_ = (None)
        item_106 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_momentum = (
            None
        )
        item_107 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_eps = (
            None
        )
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_,
            False,
            item_106,
            item_107,
        )
        conv2d_53 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_attn_modules_proj_modules_bn_parameters_bias_ = (item_106) = (
            item_107
        ) = None
        x_51 = input_21 + batch_norm_53
        input_21 = batch_norm_53 = None
        conv2d_54 = torch.conv2d(
            x_51,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        item_108 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_momentum = (
            None
        )
        item_109 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_eps = (
            None
        )
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_,
            False,
            item_108,
            item_109,
        )
        conv2d_54 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_0_modules_bn_parameters_bias_ = (item_108) = (
            item_109
        ) = None
        input_22 = torch.nn.functional.silu(batch_norm_54, inplace=True)
        batch_norm_54 = None
        conv2d_55 = torch.conv2d(
            input_22,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_conv_parameters_weight_ = (None)
        item_110 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_momentum = (
            None
        )
        item_111 = (
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps.item()
        )
        l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_eps = (
            None
        )
        input_23 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_,
            False,
            item_110,
            item_111,
        )
        conv2d_55 = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_buffers_running_var_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_weight_ = l_self_modules_model_modules_8_modules_m_modules_1_modules_1_modules_mlp_modules_1_modules_bn_parameters_bias_ = (item_110) = (
            item_111
        ) = None
        input_24 = x_51 + input_23
        x_51 = input_23 = None
        cat_3 = torch.cat([silu_19, input_18, input_24], 1)
        silu_19 = input_18 = input_24 = None
        conv2d_56 = torch.conv2d(
            cat_3,
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = (
            l_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_
        ) = None
        item_112 = l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum.item()
        l_self_modules_model_modules_8_modules_cv2_modules_bn_momentum = None
        item_113 = l_self_modules_model_modules_8_modules_cv2_modules_bn_eps.item()
        l_self_modules_model_modules_8_modules_cv2_modules_bn_eps = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_,
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_,
            False,
            item_112,
            item_113,
        )
        conv2d_56 = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_
        ) = item_112 = item_113 = None
        x_52 = torch.nn.functional.silu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        conv2d_57 = torch.conv2d(
            x_52,
            l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = (
            l_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_
        ) = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_57 = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_
        ) = (
            l_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_
        ) = None
        silu_25 = torch.nn.functional.silu(batch_norm_57, inplace=True)
        batch_norm_57 = None
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(silu_25, 1)
        silu_25 = None
        flatten_8 = adaptive_avg_pool2d.flatten(1)
        adaptive_avg_pool2d = None
        dropout = torch.nn.functional.dropout(flatten_8, 0.0, False, True)
        flatten_8 = None
        x_53 = torch._C._nn.linear(
            dropout,
            l_self_modules_model_modules_9_modules_linear_parameters_weight_,
            l_self_modules_model_modules_9_modules_linear_parameters_bias_,
        )
        dropout = (
            l_self_modules_model_modules_9_modules_linear_parameters_weight_
        ) = l_self_modules_model_modules_9_modules_linear_parameters_bias_ = None
        y = x_53.softmax(1)
        return (y, x_53)
