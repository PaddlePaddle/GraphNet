import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        s10: torch.SymInt,
        L_sensor_vals_: torch.Tensor,
        L_sensor_coords_: torch.Tensor,
        L_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_mlp1_modules_1_eps: torch.Tensor,
        L_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_attn_parameters_in_proj_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_attn_parameters_in_proj_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_attn_dropout: torch.Tensor,
        L_self_modules_sensor_encoder_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_sensor_encoder_modules_norm_eps: torch.Tensor,
        L_self_modules_projector_modules_query_mlp_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_query_mlp_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_grid_coords_: torch.Tensor,
        L_self_modules_projector_modules_query_mlp_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_query_mlp_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_projector_transition_x: torch.Tensor,
        L_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_bias_: torch.nn.parameter.Parameter,
        G_np_pi: torch.Tensor,
        L_self_modules_projector_modules_sensor_embedding_buffers_B_high_: torch.Tensor,
        L_self_modules_projector_modules_key_proj_smooth_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_key_proj_smooth_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_key_proj_detail_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_projector_modules_key_proj_detail_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_projector_scale: torch.Tensor,
        L_base_flow_: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_1_momentum: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_1_eps: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_4_momentum: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_in_conv_modules_double_conv_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_in_conv_modules_double_conv_modules_4_eps: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_momentum: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_1_eps: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_momentum: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down1_modules_1_modules_double_conv_modules_4_eps: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_momentum: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_1_eps: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_momentum: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_down2_modules_1_modules_double_conv_modules_4_eps: torch.Tensor,
        L_self_modules_up1_scale_factor: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_1_momentum: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_1_eps: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_4_momentum: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_up1_modules_double_conv_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up1_modules_double_conv_modules_4_eps: torch.Tensor,
        L_self_modules_up2_scale_factor: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_1_momentum: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_1_eps: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_4_momentum: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_conv_up2_modules_double_conv_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_up2_modules_double_conv_modules_4_eps: torch.Tensor,
        L_self_modules_out_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_out_conv_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_sensor_vals_ = L_sensor_vals_
        l_sensor_coords_ = L_sensor_coords_
        l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_weight_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_weight_
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_bias_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_bias_
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_weight_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_weight_
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_bias_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_bias_
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_1_eps = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_1_eps
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_weight_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_weight_
        )
        l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_bias_ = (
            L_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_bias_
        )
        l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_bias_ = (
            L_self_modules_sensor_encoder_modules_attn_parameters_in_proj_bias_
        )
        l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_weight_ = (
            L_self_modules_sensor_encoder_modules_attn_parameters_in_proj_weight_
        )
        l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_weight_ = L_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_weight_
        l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_bias_ = (
            L_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_bias_
        )
        l_self_modules_sensor_encoder_modules_attn_dropout = (
            L_self_modules_sensor_encoder_modules_attn_dropout
        )
        l_self_modules_sensor_encoder_modules_norm_parameters_weight_ = (
            L_self_modules_sensor_encoder_modules_norm_parameters_weight_
        )
        l_self_modules_sensor_encoder_modules_norm_parameters_bias_ = (
            L_self_modules_sensor_encoder_modules_norm_parameters_bias_
        )
        l_self_modules_sensor_encoder_modules_norm_eps = (
            L_self_modules_sensor_encoder_modules_norm_eps
        )
        l_self_modules_projector_modules_query_mlp_modules_0_parameters_weight_ = (
            L_self_modules_projector_modules_query_mlp_modules_0_parameters_weight_
        )
        l_self_modules_projector_modules_query_mlp_modules_0_parameters_bias_ = (
            L_self_modules_projector_modules_query_mlp_modules_0_parameters_bias_
        )
        l_grid_coords_ = L_grid_coords_
        l_self_modules_projector_modules_query_mlp_modules_2_parameters_weight_ = (
            L_self_modules_projector_modules_query_mlp_modules_2_parameters_weight_
        )
        l_self_modules_projector_modules_query_mlp_modules_2_parameters_bias_ = (
            L_self_modules_projector_modules_query_mlp_modules_2_parameters_bias_
        )
        l_self_modules_projector_transition_x = L_self_modules_projector_transition_x
        l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_weight_ = L_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_weight_
        l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_bias_ = L_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_bias_
        g_np_pi = G_np_pi
        l_self_modules_projector_modules_sensor_embedding_buffers_b_high_ = (
            L_self_modules_projector_modules_sensor_embedding_buffers_B_high_
        )
        l_self_modules_projector_modules_key_proj_smooth_parameters_weight_ = (
            L_self_modules_projector_modules_key_proj_smooth_parameters_weight_
        )
        l_self_modules_projector_modules_key_proj_smooth_parameters_bias_ = (
            L_self_modules_projector_modules_key_proj_smooth_parameters_bias_
        )
        l_self_modules_projector_modules_key_proj_detail_parameters_weight_ = (
            L_self_modules_projector_modules_key_proj_detail_parameters_weight_
        )
        l_self_modules_projector_modules_key_proj_detail_parameters_bias_ = (
            L_self_modules_projector_modules_key_proj_detail_parameters_bias_
        )
        l_self_modules_projector_scale = L_self_modules_projector_scale
        l_base_flow_ = L_base_flow_
        l_self_modules_in_conv_modules_double_conv_modules_0_parameters_weight_ = (
            L_self_modules_in_conv_modules_double_conv_modules_0_parameters_weight_
        )
        l_self_modules_in_conv_modules_double_conv_modules_0_parameters_bias_ = (
            L_self_modules_in_conv_modules_double_conv_modules_0_parameters_bias_
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_momentum = (
            L_self_modules_in_conv_modules_double_conv_modules_1_momentum
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_mean_ = (
            L_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_mean_
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_var_ = (
            L_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_var_
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_parameters_weight_ = (
            L_self_modules_in_conv_modules_double_conv_modules_1_parameters_weight_
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_parameters_bias_ = (
            L_self_modules_in_conv_modules_double_conv_modules_1_parameters_bias_
        )
        l_self_modules_in_conv_modules_double_conv_modules_1_eps = (
            L_self_modules_in_conv_modules_double_conv_modules_1_eps
        )
        l_self_modules_in_conv_modules_double_conv_modules_3_parameters_weight_ = (
            L_self_modules_in_conv_modules_double_conv_modules_3_parameters_weight_
        )
        l_self_modules_in_conv_modules_double_conv_modules_3_parameters_bias_ = (
            L_self_modules_in_conv_modules_double_conv_modules_3_parameters_bias_
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_momentum = (
            L_self_modules_in_conv_modules_double_conv_modules_4_momentum
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_mean_ = (
            L_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_mean_
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_var_ = (
            L_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_var_
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_parameters_weight_ = (
            L_self_modules_in_conv_modules_double_conv_modules_4_parameters_weight_
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_parameters_bias_ = (
            L_self_modules_in_conv_modules_double_conv_modules_4_parameters_bias_
        )
        l_self_modules_in_conv_modules_double_conv_modules_4_eps = (
            L_self_modules_in_conv_modules_double_conv_modules_4_eps
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_weight_ = L_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_weight_
        l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_bias_ = L_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_bias_
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_momentum = (
            L_self_modules_down1_modules_1_modules_double_conv_modules_1_momentum
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_mean_ = L_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_mean_
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_var_ = L_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_var_
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_weight_ = L_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_weight_
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_bias_ = L_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_bias_
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_eps = (
            L_self_modules_down1_modules_1_modules_double_conv_modules_1_eps
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_weight_ = L_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_weight_
        l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_bias_ = L_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_bias_
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_momentum = (
            L_self_modules_down1_modules_1_modules_double_conv_modules_4_momentum
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_mean_ = L_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_mean_
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_var_ = L_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_var_
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_weight_ = L_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_weight_
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_bias_ = L_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_bias_
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_eps = (
            L_self_modules_down1_modules_1_modules_double_conv_modules_4_eps
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_weight_ = L_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_weight_
        l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_bias_ = L_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_bias_
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_momentum = (
            L_self_modules_down2_modules_1_modules_double_conv_modules_1_momentum
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_mean_ = L_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_mean_
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_var_ = L_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_var_
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_weight_ = L_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_weight_
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_bias_ = L_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_bias_
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_eps = (
            L_self_modules_down2_modules_1_modules_double_conv_modules_1_eps
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_weight_ = L_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_weight_
        l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_bias_ = L_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_bias_
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_momentum = (
            L_self_modules_down2_modules_1_modules_double_conv_modules_4_momentum
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_mean_ = L_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_mean_
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_var_ = L_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_var_
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_weight_ = L_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_weight_
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_bias_ = L_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_bias_
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_eps = (
            L_self_modules_down2_modules_1_modules_double_conv_modules_4_eps
        )
        l_self_modules_up1_scale_factor = L_self_modules_up1_scale_factor
        l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_weight_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_0_parameters_weight_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_bias_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_0_parameters_bias_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_momentum = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_momentum
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_mean_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_mean_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_var_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_var_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_weight_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_parameters_weight_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_bias_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_parameters_bias_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_1_eps = (
            L_self_modules_conv_up1_modules_double_conv_modules_1_eps
        )
        l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_weight_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_3_parameters_weight_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_bias_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_3_parameters_bias_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_momentum = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_momentum
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_mean_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_mean_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_var_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_var_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_weight_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_parameters_weight_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_bias_ = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_parameters_bias_
        )
        l_self_modules_conv_up1_modules_double_conv_modules_4_eps = (
            L_self_modules_conv_up1_modules_double_conv_modules_4_eps
        )
        l_self_modules_up2_scale_factor = L_self_modules_up2_scale_factor
        l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_weight_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_0_parameters_weight_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_bias_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_0_parameters_bias_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_momentum = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_momentum
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_mean_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_mean_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_var_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_var_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_weight_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_parameters_weight_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_bias_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_parameters_bias_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_1_eps = (
            L_self_modules_conv_up2_modules_double_conv_modules_1_eps
        )
        l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_weight_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_3_parameters_weight_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_bias_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_3_parameters_bias_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_momentum = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_momentum
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_mean_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_mean_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_var_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_var_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_weight_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_parameters_weight_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_bias_ = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_parameters_bias_
        )
        l_self_modules_conv_up2_modules_double_conv_modules_4_eps = (
            L_self_modules_conv_up2_modules_double_conv_modules_4_eps
        )
        l_self_modules_out_conv_parameters_weight_ = (
            L_self_modules_out_conv_parameters_weight_
        )
        l_self_modules_out_conv_parameters_bias_ = (
            L_self_modules_out_conv_parameters_bias_
        )
        node_in = torch.cat([l_sensor_vals_, l_sensor_coords_], dim=-1)
        l_sensor_vals_ = None
        input_1 = torch.nn.functional.linear(
            node_in,
            l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_weight_,
            l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_bias_,
        )
        node_in = (
            l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_weight_
        ) = l_self_modules_sensor_encoder_modules_mlp1_modules_0_parameters_bias_ = None
        item = l_self_modules_sensor_encoder_modules_mlp1_modules_1_eps.item()
        l_self_modules_sensor_encoder_modules_mlp1_modules_1_eps = None
        input_2 = torch.nn.functional.layer_norm(
            input_1,
            (64,),
            l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_weight_,
            l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_bias_,
            item,
        )
        input_1 = (
            l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_weight_
        ) = (
            l_self_modules_sensor_encoder_modules_mlp1_modules_1_parameters_bias_
        ) = item = None
        input_3 = torch.nn.functional.silu(input_2, inplace=False)
        input_2 = None
        input_4 = torch.nn.functional.linear(
            input_3,
            l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_weight_,
            l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_bias_,
        )
        input_3 = (
            l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_weight_
        ) = l_self_modules_sensor_encoder_modules_mlp1_modules_3_parameters_bias_ = None
        query = input_4.transpose(1, 0)
        item_1 = l_self_modules_sensor_encoder_modules_attn_dropout.item()
        l_self_modules_sensor_encoder_modules_attn_dropout = None
        multi_head_attention_forward = torch.nn.functional.multi_head_attention_forward(
            query,
            query,
            query,
            64,
            4,
            l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_weight_,
            l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_bias_,
            None,
            None,
            False,
            item_1,
            l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_weight_,
            l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_bias_,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        )
        query = (
            l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_weight_
        ) = (
            l_self_modules_sensor_encoder_modules_attn_parameters_in_proj_bias_
        ) = (
            item_1
        ) = l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_weight_ = (
            l_self_modules_sensor_encoder_modules_attn_modules_out_proj_parameters_bias_
        ) = None
        attn_output = multi_head_attention_forward[0]
        multi_head_attention_forward = None
        h_attn = attn_output.transpose(1, 0)
        attn_output = None
        add = input_4 + h_attn
        input_4 = h_attn = None
        item_2 = l_self_modules_sensor_encoder_modules_norm_eps.item()
        l_self_modules_sensor_encoder_modules_norm_eps = None
        h = torch.nn.functional.layer_norm(
            add,
            (64,),
            l_self_modules_sensor_encoder_modules_norm_parameters_weight_,
            l_self_modules_sensor_encoder_modules_norm_parameters_bias_,
            item_2,
        )
        add = (
            l_self_modules_sensor_encoder_modules_norm_parameters_weight_
        ) = l_self_modules_sensor_encoder_modules_norm_parameters_bias_ = item_2 = None
        global_context = torch.mean(h, dim=1)
        input_5 = torch.nn.functional.linear(
            l_grid_coords_,
            l_self_modules_projector_modules_query_mlp_modules_0_parameters_weight_,
            l_self_modules_projector_modules_query_mlp_modules_0_parameters_bias_,
        )
        l_self_modules_projector_modules_query_mlp_modules_0_parameters_weight_ = (
            l_self_modules_projector_modules_query_mlp_modules_0_parameters_bias_
        ) = None
        input_6 = torch.nn.functional.silu(input_5, inplace=False)
        input_5 = None
        input_7 = torch.nn.functional.linear(
            input_6,
            l_self_modules_projector_modules_query_mlp_modules_2_parameters_weight_,
            l_self_modules_projector_modules_query_mlp_modules_2_parameters_bias_,
        )
        input_6 = (
            l_self_modules_projector_modules_query_mlp_modules_2_parameters_weight_
        ) = l_self_modules_projector_modules_query_mlp_modules_2_parameters_bias_ = None
        Q_smooth = input_7[(Ellipsis, slice(None, 64, None))]
        Q_detail = input_7[(Ellipsis, slice(64, None, None))]
        input_7 = None
        grid_x = l_grid_coords_[(Ellipsis, slice(0, 1, None))]
        item_3 = l_self_modules_projector_transition_x.item()
        l_self_modules_projector_transition_x = None
        sub = grid_x - item_3
        grid_x = item_3 = None
        mul = 50.0 * sub
        sub = None
        beta = torch.sigmoid(mul)
        mul = None
        Q_detail_1 = Q_detail * beta
        Q_detail = beta = None
        linear_4 = torch.nn.functional.linear(
            l_sensor_coords_,
            l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_weight_,
            l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_bias_,
        )
        l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_weight_ = l_self_modules_projector_modules_sensor_embedding_modules_raw_proj_parameters_bias_ = (None)
        feat_smooth = torch.nn.functional.silu(linear_4)
        linear_4 = None
        item_4 = g_np_pi.item()
        g_np_pi = None
        mul_2 = 2 * item_4
        item_4 = None
        mul_3 = mul_2 * l_sensor_coords_
        mul_2 = l_sensor_coords_ = None
        proj_high = (
            mul_3 @ l_self_modules_projector_modules_sensor_embedding_buffers_b_high_
        )
        mul_3 = l_self_modules_projector_modules_sensor_embedding_buffers_b_high_ = None
        sin = torch.sin(proj_high)
        cos = torch.cos(proj_high)
        proj_high = None
        feat_detail = torch.cat([sin, cos], dim=-1)
        sin = cos = None
        K_smooth = torch.nn.functional.linear(
            feat_smooth,
            l_self_modules_projector_modules_key_proj_smooth_parameters_weight_,
            l_self_modules_projector_modules_key_proj_smooth_parameters_bias_,
        )
        feat_smooth = (
            l_self_modules_projector_modules_key_proj_smooth_parameters_weight_
        ) = l_self_modules_projector_modules_key_proj_smooth_parameters_bias_ = None
        K_detail = torch.nn.functional.linear(
            feat_detail,
            l_self_modules_projector_modules_key_proj_detail_parameters_weight_,
            l_self_modules_projector_modules_key_proj_detail_parameters_bias_,
        )
        feat_detail = (
            l_self_modules_projector_modules_key_proj_detail_parameters_weight_
        ) = l_self_modules_projector_modules_key_proj_detail_parameters_bias_ = None
        score_smooth = torch.functional.einsum("bhwc,bsc->bhws", Q_smooth, K_smooth)
        Q_smooth = K_smooth = None
        score_detail = torch.functional.einsum("bhwc,bsc->bhws", Q_detail_1, K_detail)
        Q_detail_1 = K_detail = None
        add_1 = score_smooth + score_detail
        score_smooth = score_detail = None
        item_5 = l_self_modules_projector_scale.item()
        l_self_modules_projector_scale = None
        attn_logits = add_1 * item_5
        add_1 = item_5 = None
        attn = torch.nn.functional.softmax(attn_logits, dim=-1)
        attn_logits = None
        out = torch.functional.einsum("bhws,bsc->bhwc", attn, h)
        attn = h = None
        grid_feats = out.permute(0, 3, 1, 2)
        out = None
        view = global_context.view(1, 64, 1, 1)
        global_context = None
        global_map = view.expand(1, 64, 128, 256)
        view = None
        coords_map = l_grid_coords_.permute(0, 3, 1, 2)
        l_grid_coords_ = None
        coords_map_1 = torch.nn.functional.interpolate(
            coords_map, size=(128, 256), mode="bilinear", align_corners=True
        )
        coords_map = None
        base_flow = torch.nn.functional.interpolate(
            l_base_flow_, size=(128, 256), mode="bilinear", align_corners=True
        )
        l_base_flow_ = None
        x = torch.cat([grid_feats, coords_map_1, global_map, base_flow], dim=1)
        grid_feats = coords_map_1 = global_map = None
        input_8 = torch.conv2d(
            x,
            l_self_modules_in_conv_modules_double_conv_modules_0_parameters_weight_,
            l_self_modules_in_conv_modules_double_conv_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_in_conv_modules_double_conv_modules_0_parameters_weight_
        ) = l_self_modules_in_conv_modules_double_conv_modules_0_parameters_bias_ = None
        item_6 = l_self_modules_in_conv_modules_double_conv_modules_1_momentum.item()
        l_self_modules_in_conv_modules_double_conv_modules_1_momentum = None
        item_7 = l_self_modules_in_conv_modules_double_conv_modules_1_eps.item()
        l_self_modules_in_conv_modules_double_conv_modules_1_eps = None
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_mean_,
            l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_var_,
            l_self_modules_in_conv_modules_double_conv_modules_1_parameters_weight_,
            l_self_modules_in_conv_modules_double_conv_modules_1_parameters_bias_,
            False,
            item_6,
            item_7,
        )
        input_8 = (
            l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_1_buffers_running_var_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_1_parameters_weight_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_1_parameters_bias_
        ) = item_6 = item_7 = None
        input_10 = torch.nn.functional.silu(input_9, inplace=True)
        input_9 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_in_conv_modules_double_conv_modules_3_parameters_weight_,
            l_self_modules_in_conv_modules_double_conv_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = (
            l_self_modules_in_conv_modules_double_conv_modules_3_parameters_weight_
        ) = l_self_modules_in_conv_modules_double_conv_modules_3_parameters_bias_ = None
        item_8 = l_self_modules_in_conv_modules_double_conv_modules_4_momentum.item()
        l_self_modules_in_conv_modules_double_conv_modules_4_momentum = None
        item_9 = l_self_modules_in_conv_modules_double_conv_modules_4_eps.item()
        l_self_modules_in_conv_modules_double_conv_modules_4_eps = None
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_mean_,
            l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_var_,
            l_self_modules_in_conv_modules_double_conv_modules_4_parameters_weight_,
            l_self_modules_in_conv_modules_double_conv_modules_4_parameters_bias_,
            False,
            item_8,
            item_9,
        )
        input_11 = (
            l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_4_buffers_running_var_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_4_parameters_weight_
        ) = (
            l_self_modules_in_conv_modules_double_conv_modules_4_parameters_bias_
        ) = item_8 = item_9 = None
        input_13 = torch.nn.functional.silu(input_12, inplace=True)
        input_12 = None
        input_14 = torch.nn.functional.max_pool2d(
            input_13, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_weight_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_weight_ = l_self_modules_down1_modules_1_modules_double_conv_modules_0_parameters_bias_ = (None)
        item_10 = (
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_momentum.item()
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_momentum = None
        item_11 = (
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_eps.item()
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_1_eps = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_mean_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_var_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_weight_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_bias_,
            False,
            item_10,
            item_11,
        )
        input_15 = l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_mean_ = l_self_modules_down1_modules_1_modules_double_conv_modules_1_buffers_running_var_ = l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_weight_ = l_self_modules_down1_modules_1_modules_double_conv_modules_1_parameters_bias_ = (item_10) = (
            item_11
        ) = None
        input_17 = torch.nn.functional.silu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_weight_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_weight_ = l_self_modules_down1_modules_1_modules_double_conv_modules_3_parameters_bias_ = (None)
        item_12 = (
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_momentum.item()
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_momentum = None
        item_13 = (
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_eps.item()
        )
        l_self_modules_down1_modules_1_modules_double_conv_modules_4_eps = None
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_mean_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_var_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_weight_,
            l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_bias_,
            False,
            item_12,
            item_13,
        )
        input_18 = l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_mean_ = l_self_modules_down1_modules_1_modules_double_conv_modules_4_buffers_running_var_ = l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_weight_ = l_self_modules_down1_modules_1_modules_double_conv_modules_4_parameters_bias_ = (item_12) = (
            item_13
        ) = None
        input_20 = torch.nn.functional.silu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.nn.functional.max_pool2d(
            input_20, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        input_22 = torch.conv2d(
            input_21,
            l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_weight_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_weight_ = l_self_modules_down2_modules_1_modules_double_conv_modules_0_parameters_bias_ = (None)
        item_14 = (
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_momentum.item()
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_momentum = None
        item_15 = (
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_eps.item()
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_1_eps = None
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_mean_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_var_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_weight_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_bias_,
            False,
            item_14,
            item_15,
        )
        input_22 = l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_mean_ = l_self_modules_down2_modules_1_modules_double_conv_modules_1_buffers_running_var_ = l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_weight_ = l_self_modules_down2_modules_1_modules_double_conv_modules_1_parameters_bias_ = (item_14) = (
            item_15
        ) = None
        input_24 = torch.nn.functional.silu(input_23, inplace=True)
        input_23 = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_weight_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_weight_ = l_self_modules_down2_modules_1_modules_double_conv_modules_3_parameters_bias_ = (None)
        item_16 = (
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_momentum.item()
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_momentum = None
        item_17 = (
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_eps.item()
        )
        l_self_modules_down2_modules_1_modules_double_conv_modules_4_eps = None
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_mean_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_var_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_weight_,
            l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_bias_,
            False,
            item_16,
            item_17,
        )
        input_25 = l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_mean_ = l_self_modules_down2_modules_1_modules_double_conv_modules_4_buffers_running_var_ = l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_weight_ = l_self_modules_down2_modules_1_modules_double_conv_modules_4_parameters_bias_ = (item_16) = (
            item_17
        ) = None
        input_27 = torch.nn.functional.silu(input_26, inplace=True)
        input_26 = None
        item_18 = l_self_modules_up1_scale_factor.item()
        l_self_modules_up1_scale_factor = None
        x_1 = torch.nn.functional.interpolate(
            input_27, None, item_18, "bilinear", True, recompute_scale_factor=None
        )
        input_27 = item_18 = None
        x_2 = torch.cat([x_1, input_20], dim=1)
        x_1 = input_20 = None
        input_28 = torch.conv2d(
            x_2,
            l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_weight_,
            l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = (
            l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_weight_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_0_parameters_bias_
        ) = None
        item_19 = l_self_modules_conv_up1_modules_double_conv_modules_1_momentum.item()
        l_self_modules_conv_up1_modules_double_conv_modules_1_momentum = None
        item_20 = l_self_modules_conv_up1_modules_double_conv_modules_1_eps.item()
        l_self_modules_conv_up1_modules_double_conv_modules_1_eps = None
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_mean_,
            l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_var_,
            l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_weight_,
            l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_bias_,
            False,
            item_19,
            item_20,
        )
        input_28 = (
            l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_weight_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_1_parameters_bias_
        ) = item_19 = item_20 = None
        input_30 = torch.nn.functional.silu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_weight_,
            l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_30 = (
            l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_3_parameters_bias_
        ) = None
        item_21 = l_self_modules_conv_up1_modules_double_conv_modules_4_momentum.item()
        l_self_modules_conv_up1_modules_double_conv_modules_4_momentum = None
        item_22 = l_self_modules_conv_up1_modules_double_conv_modules_4_eps.item()
        l_self_modules_conv_up1_modules_double_conv_modules_4_eps = None
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_mean_,
            l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_var_,
            l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_weight_,
            l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_bias_,
            False,
            item_21,
            item_22,
        )
        input_31 = (
            l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_4_buffers_running_var_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_weight_
        ) = (
            l_self_modules_conv_up1_modules_double_conv_modules_4_parameters_bias_
        ) = item_21 = item_22 = None
        input_33 = torch.nn.functional.silu(input_32, inplace=True)
        input_32 = None
        item_23 = l_self_modules_up2_scale_factor.item()
        l_self_modules_up2_scale_factor = None
        x_3 = torch.nn.functional.interpolate(
            input_33, None, item_23, "bilinear", True, recompute_scale_factor=None
        )
        input_33 = item_23 = None
        x_4 = torch.cat([x_3, input_13], dim=1)
        x_3 = input_13 = None
        input_34 = torch.conv2d(
            x_4,
            l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_weight_,
            l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = (
            l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_weight_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_0_parameters_bias_
        ) = None
        item_24 = l_self_modules_conv_up2_modules_double_conv_modules_1_momentum.item()
        l_self_modules_conv_up2_modules_double_conv_modules_1_momentum = None
        item_25 = l_self_modules_conv_up2_modules_double_conv_modules_1_eps.item()
        l_self_modules_conv_up2_modules_double_conv_modules_1_eps = None
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_mean_,
            l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_var_,
            l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_weight_,
            l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_bias_,
            False,
            item_24,
            item_25,
        )
        input_34 = (
            l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_weight_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_1_parameters_bias_
        ) = item_24 = item_25 = None
        input_36 = torch.nn.functional.silu(input_35, inplace=True)
        input_35 = None
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_weight_,
            l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_36 = (
            l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_3_parameters_bias_
        ) = None
        item_26 = l_self_modules_conv_up2_modules_double_conv_modules_4_momentum.item()
        l_self_modules_conv_up2_modules_double_conv_modules_4_momentum = None
        item_27 = l_self_modules_conv_up2_modules_double_conv_modules_4_eps.item()
        l_self_modules_conv_up2_modules_double_conv_modules_4_eps = None
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_mean_,
            l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_var_,
            l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_weight_,
            l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_bias_,
            False,
            item_26,
            item_27,
        )
        input_37 = (
            l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_4_buffers_running_var_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_weight_
        ) = (
            l_self_modules_conv_up2_modules_double_conv_modules_4_parameters_bias_
        ) = item_26 = item_27 = None
        input_39 = torch.nn.functional.silu(input_38, inplace=True)
        input_38 = None
        fluc_out = torch.conv2d(
            input_39,
            l_self_modules_out_conv_parameters_weight_,
            l_self_modules_out_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = (
            l_self_modules_out_conv_parameters_weight_
        ) = l_self_modules_out_conv_parameters_bias_ = None
        add_2 = fluc_out + base_flow
        fluc_out = base_flow = None
        return (add_2,)
