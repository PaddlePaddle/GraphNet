import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_modules_rope_buffers_pos_embed_: torch.Tensor,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_patch_embed_modules_proj_parameters_weight_ = (
            L_self_modules_patch_embed_modules_proj_parameters_weight_
        )
        l_self_modules_patch_embed_modules_proj_parameters_bias_ = (
            L_self_modules_patch_embed_modules_proj_parameters_bias_
        )
        l_self_parameters_pos_embed_ = L_self_parameters_pos_embed_
        l_self_modules_rope_buffers_pos_embed_ = L_self_modules_rope_buffers_pos_embed_
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_proj_parameters_bias_,
            (16, 16),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_patch_embed_modules_proj_parameters_weight_
        ) = l_self_modules_patch_embed_modules_proj_parameters_bias_ = None
        flatten = x.flatten(2)
        x = None
        x_1 = flatten.transpose(1, 2)
        flatten = None
        expand = l_self_parameters_cls_token_.expand(1, -1, -1)
        l_self_parameters_cls_token_ = None
        x_2 = torch.cat([expand, x_1], dim=1)
        expand = x_1 = None
        x_3 = x_2 + l_self_parameters_pos_embed_
        x_2 = l_self_parameters_pos_embed_ = None
        x_4 = torch.nn.functional.dropout(x_3, 0.0, False, False)
        x_3 = None
        x_5 = torch.nn.functional.layer_norm(
            x_4,
            (768,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        ) = None
        linear = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape = linear.reshape(1, 197, 12, -1)
        linear = None
        q = reshape.transpose(1, 2)
        reshape = None
        linear_1 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_1 = linear_1.reshape(1, 197, 12, -1)
        linear_1 = None
        k = reshape_1.transpose(1, 2)
        reshape_1 = None
        linear_2 = torch._C._nn.linear(
            x_5,
            l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_2 = linear_2.reshape(1, 197, 12, -1)
        linear_2 = None
        v = reshape_2.transpose(1, 2)
        reshape_2 = None
        getitem_4 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_5 = q[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q = None
        tensor_split = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb = tensor_split[0]
        cos_emb = tensor_split[1]
        tensor_split = None
        mul = getitem_5 * cos_emb
        cos_emb = None
        getitem_8 = getitem_5[(Ellipsis, slice(1, None, 2))]
        neg = -getitem_8
        getitem_8 = None
        getitem_9 = getitem_5[(Ellipsis, slice(None, None, 2))]
        getitem_5 = None
        stack = torch.stack([neg, getitem_9], -1)
        neg = getitem_9 = None
        reshape_3 = stack.reshape((1, 12, 196, 64))
        stack = None
        mul_1 = reshape_3 * sin_emb
        reshape_3 = sin_emb = None
        add_1 = mul + mul_1
        mul = mul_1 = None
        cat_1 = torch.cat([getitem_4, add_1], dim=2)
        getitem_4 = add_1 = None
        q_1 = cat_1.type_as(v)
        cat_1 = None
        getitem_10 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_11 = k[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k = None
        tensor_split_1 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_1 = tensor_split_1[0]
        cos_emb_1 = tensor_split_1[1]
        tensor_split_1 = None
        mul_2 = getitem_11 * cos_emb_1
        cos_emb_1 = None
        getitem_14 = getitem_11[(Ellipsis, slice(1, None, 2))]
        neg_1 = -getitem_14
        getitem_14 = None
        getitem_15 = getitem_11[(Ellipsis, slice(None, None, 2))]
        getitem_11 = None
        stack_1 = torch.stack([neg_1, getitem_15], -1)
        neg_1 = getitem_15 = None
        reshape_4 = stack_1.reshape((1, 12, 196, 64))
        stack_1 = None
        mul_3 = reshape_4 * sin_emb_1
        reshape_4 = sin_emb_1 = None
        add_2 = mul_2 + mul_3
        mul_2 = mul_3 = None
        cat_2 = torch.cat([getitem_10, add_2], dim=2)
        getitem_10 = add_2 = None
        k_1 = cat_2.type_as(v)
        cat_2 = None
        x_6 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v = None
        transpose_4 = x_6.transpose(1, 2)
        x_6 = None
        x_7 = transpose_4.reshape(1, 197, 768)
        transpose_4 = None
        x_8 = torch.nn.functional.layer_norm(
            x_7,
            (768,),
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_9 = torch._C._nn.linear(
            x_8,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_8 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_10 = torch.nn.functional.dropout(x_9, 0.0, False, False)
        x_9 = None
        x_11 = x_4 + x_10
        x_4 = x_10 = None
        x_12 = torch.nn.functional.layer_norm(
            x_11,
            (768,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        ) = None
        x_gate = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_12 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu = torch.nn.functional.silu(x_gate, inplace=False)
        x_gate = None
        x_14 = silu * x_13
        silu = x_13 = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (2048,),
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_15 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_17 = torch._C._nn.linear(
            x_16,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_16 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.dropout(x_17, 0.0, False, False)
        x_17 = None
        x_19 = x_11 + x_18
        x_11 = x_18 = None
        x_20 = torch.nn.functional.layer_norm(
            x_19,
            (768,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        ) = None
        linear_7 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_6 = linear_7.reshape(1, 197, 12, -1)
        linear_7 = None
        q_2 = reshape_6.transpose(1, 2)
        reshape_6 = None
        linear_8 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_7 = linear_8.reshape(1, 197, 12, -1)
        linear_8 = None
        k_2 = reshape_7.transpose(1, 2)
        reshape_7 = None
        linear_9 = torch._C._nn.linear(
            x_20,
            l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_8 = linear_9.reshape(1, 197, 12, -1)
        linear_9 = None
        v_1 = reshape_8.transpose(1, 2)
        reshape_8 = None
        getitem_16 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_17 = q_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_2 = None
        tensor_split_2 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_2 = tensor_split_2[0]
        cos_emb_2 = tensor_split_2[1]
        tensor_split_2 = None
        mul_5 = getitem_17 * cos_emb_2
        cos_emb_2 = None
        getitem_20 = getitem_17[(Ellipsis, slice(1, None, 2))]
        neg_2 = -getitem_20
        getitem_20 = None
        getitem_21 = getitem_17[(Ellipsis, slice(None, None, 2))]
        getitem_17 = None
        stack_2 = torch.stack([neg_2, getitem_21], -1)
        neg_2 = getitem_21 = None
        reshape_9 = stack_2.reshape((1, 12, 196, 64))
        stack_2 = None
        mul_6 = reshape_9 * sin_emb_2
        reshape_9 = sin_emb_2 = None
        add_5 = mul_5 + mul_6
        mul_5 = mul_6 = None
        cat_3 = torch.cat([getitem_16, add_5], dim=2)
        getitem_16 = add_5 = None
        q_3 = cat_3.type_as(v_1)
        cat_3 = None
        getitem_22 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_23 = k_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_2 = None
        tensor_split_3 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_3 = tensor_split_3[0]
        cos_emb_3 = tensor_split_3[1]
        tensor_split_3 = None
        mul_7 = getitem_23 * cos_emb_3
        cos_emb_3 = None
        getitem_26 = getitem_23[(Ellipsis, slice(1, None, 2))]
        neg_3 = -getitem_26
        getitem_26 = None
        getitem_27 = getitem_23[(Ellipsis, slice(None, None, 2))]
        getitem_23 = None
        stack_3 = torch.stack([neg_3, getitem_27], -1)
        neg_3 = getitem_27 = None
        reshape_10 = stack_3.reshape((1, 12, 196, 64))
        stack_3 = None
        mul_8 = reshape_10 * sin_emb_3
        reshape_10 = sin_emb_3 = None
        add_6 = mul_7 + mul_8
        mul_7 = mul_8 = None
        cat_4 = torch.cat([getitem_22, add_6], dim=2)
        getitem_22 = add_6 = None
        k_3 = cat_4.type_as(v_1)
        cat_4 = None
        x_21 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_1, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_1 = None
        transpose_8 = x_21.transpose(1, 2)
        x_21 = None
        x_22 = transpose_8.reshape(1, 197, 768)
        transpose_8 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (768,),
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_22 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_23 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_25 = torch.nn.functional.dropout(x_24, 0.0, False, False)
        x_24 = None
        x_26 = x_19 + x_25
        x_19 = x_25 = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (768,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        ) = None
        x_gate_1 = torch._C._nn.linear(
            x_27,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_27 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_1 = torch.nn.functional.silu(x_gate_1, inplace=False)
        x_gate_1 = None
        x_29 = silu_1 * x_28
        silu_1 = x_28 = None
        x_30 = torch.nn.functional.dropout(x_29, 0.0, False, False)
        x_29 = None
        x_31 = torch.nn.functional.layer_norm(
            x_30,
            (2048,),
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_30 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_32 = torch._C._nn.linear(
            x_31,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_31 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_33 = torch.nn.functional.dropout(x_32, 0.0, False, False)
        x_32 = None
        x_34 = x_26 + x_33
        x_26 = x_33 = None
        x_35 = torch.nn.functional.layer_norm(
            x_34,
            (768,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        ) = None
        linear_14 = torch._C._nn.linear(
            x_35,
            l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_12 = linear_14.reshape(1, 197, 12, -1)
        linear_14 = None
        q_4 = reshape_12.transpose(1, 2)
        reshape_12 = None
        linear_15 = torch._C._nn.linear(
            x_35,
            l_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_13 = linear_15.reshape(1, 197, 12, -1)
        linear_15 = None
        k_4 = reshape_13.transpose(1, 2)
        reshape_13 = None
        linear_16 = torch._C._nn.linear(
            x_35,
            l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_14 = linear_16.reshape(1, 197, 12, -1)
        linear_16 = None
        v_2 = reshape_14.transpose(1, 2)
        reshape_14 = None
        getitem_28 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_29 = q_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_4 = None
        tensor_split_4 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_4 = tensor_split_4[0]
        cos_emb_4 = tensor_split_4[1]
        tensor_split_4 = None
        mul_10 = getitem_29 * cos_emb_4
        cos_emb_4 = None
        getitem_32 = getitem_29[(Ellipsis, slice(1, None, 2))]
        neg_4 = -getitem_32
        getitem_32 = None
        getitem_33 = getitem_29[(Ellipsis, slice(None, None, 2))]
        getitem_29 = None
        stack_4 = torch.stack([neg_4, getitem_33], -1)
        neg_4 = getitem_33 = None
        reshape_15 = stack_4.reshape((1, 12, 196, 64))
        stack_4 = None
        mul_11 = reshape_15 * sin_emb_4
        reshape_15 = sin_emb_4 = None
        add_9 = mul_10 + mul_11
        mul_10 = mul_11 = None
        cat_5 = torch.cat([getitem_28, add_9], dim=2)
        getitem_28 = add_9 = None
        q_5 = cat_5.type_as(v_2)
        cat_5 = None
        getitem_34 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_35 = k_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_4 = None
        tensor_split_5 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_5 = tensor_split_5[0]
        cos_emb_5 = tensor_split_5[1]
        tensor_split_5 = None
        mul_12 = getitem_35 * cos_emb_5
        cos_emb_5 = None
        getitem_38 = getitem_35[(Ellipsis, slice(1, None, 2))]
        neg_5 = -getitem_38
        getitem_38 = None
        getitem_39 = getitem_35[(Ellipsis, slice(None, None, 2))]
        getitem_35 = None
        stack_5 = torch.stack([neg_5, getitem_39], -1)
        neg_5 = getitem_39 = None
        reshape_16 = stack_5.reshape((1, 12, 196, 64))
        stack_5 = None
        mul_13 = reshape_16 * sin_emb_5
        reshape_16 = sin_emb_5 = None
        add_10 = mul_12 + mul_13
        mul_12 = mul_13 = None
        cat_6 = torch.cat([getitem_34, add_10], dim=2)
        getitem_34 = add_10 = None
        k_5 = cat_6.type_as(v_2)
        cat_6 = None
        x_36 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_2, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_2 = None
        transpose_12 = x_36.transpose(1, 2)
        x_36 = None
        x_37 = transpose_12.reshape(1, 197, 768)
        transpose_12 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (768,),
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_39 = torch._C._nn.linear(
            x_38,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.dropout(x_39, 0.0, False, False)
        x_39 = None
        x_41 = x_34 + x_40
        x_34 = x_40 = None
        x_42 = torch.nn.functional.layer_norm(
            x_41,
            (768,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        ) = None
        x_gate_2 = torch._C._nn.linear(
            x_42,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_43 = torch._C._nn.linear(
            x_42,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_42 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_2 = torch.nn.functional.silu(x_gate_2, inplace=False)
        x_gate_2 = None
        x_44 = silu_2 * x_43
        silu_2 = x_43 = None
        x_45 = torch.nn.functional.dropout(x_44, 0.0, False, False)
        x_44 = None
        x_46 = torch.nn.functional.layer_norm(
            x_45,
            (2048,),
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_45 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_47 = torch._C._nn.linear(
            x_46,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_46 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_48 = torch.nn.functional.dropout(x_47, 0.0, False, False)
        x_47 = None
        x_49 = x_41 + x_48
        x_41 = x_48 = None
        x_50 = torch.nn.functional.layer_norm(
            x_49,
            (768,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        ) = None
        linear_21 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_18 = linear_21.reshape(1, 197, 12, -1)
        linear_21 = None
        q_6 = reshape_18.transpose(1, 2)
        reshape_18 = None
        linear_22 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_19 = linear_22.reshape(1, 197, 12, -1)
        linear_22 = None
        k_6 = reshape_19.transpose(1, 2)
        reshape_19 = None
        linear_23 = torch._C._nn.linear(
            x_50,
            l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_50 = l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_20 = linear_23.reshape(1, 197, 12, -1)
        linear_23 = None
        v_3 = reshape_20.transpose(1, 2)
        reshape_20 = None
        getitem_40 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_41 = q_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_6 = None
        tensor_split_6 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_6 = tensor_split_6[0]
        cos_emb_6 = tensor_split_6[1]
        tensor_split_6 = None
        mul_15 = getitem_41 * cos_emb_6
        cos_emb_6 = None
        getitem_44 = getitem_41[(Ellipsis, slice(1, None, 2))]
        neg_6 = -getitem_44
        getitem_44 = None
        getitem_45 = getitem_41[(Ellipsis, slice(None, None, 2))]
        getitem_41 = None
        stack_6 = torch.stack([neg_6, getitem_45], -1)
        neg_6 = getitem_45 = None
        reshape_21 = stack_6.reshape((1, 12, 196, 64))
        stack_6 = None
        mul_16 = reshape_21 * sin_emb_6
        reshape_21 = sin_emb_6 = None
        add_13 = mul_15 + mul_16
        mul_15 = mul_16 = None
        cat_7 = torch.cat([getitem_40, add_13], dim=2)
        getitem_40 = add_13 = None
        q_7 = cat_7.type_as(v_3)
        cat_7 = None
        getitem_46 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_47 = k_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_6 = None
        tensor_split_7 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_7 = tensor_split_7[0]
        cos_emb_7 = tensor_split_7[1]
        tensor_split_7 = None
        mul_17 = getitem_47 * cos_emb_7
        cos_emb_7 = None
        getitem_50 = getitem_47[(Ellipsis, slice(1, None, 2))]
        neg_7 = -getitem_50
        getitem_50 = None
        getitem_51 = getitem_47[(Ellipsis, slice(None, None, 2))]
        getitem_47 = None
        stack_7 = torch.stack([neg_7, getitem_51], -1)
        neg_7 = getitem_51 = None
        reshape_22 = stack_7.reshape((1, 12, 196, 64))
        stack_7 = None
        mul_18 = reshape_22 * sin_emb_7
        reshape_22 = sin_emb_7 = None
        add_14 = mul_17 + mul_18
        mul_17 = mul_18 = None
        cat_8 = torch.cat([getitem_46, add_14], dim=2)
        getitem_46 = add_14 = None
        k_7 = cat_8.type_as(v_3)
        cat_8 = None
        x_51 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_3, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_3 = None
        transpose_16 = x_51.transpose(1, 2)
        x_51 = None
        x_52 = transpose_16.reshape(1, 197, 768)
        transpose_16 = None
        x_53 = torch.nn.functional.layer_norm(
            x_52,
            (768,),
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_52 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_53 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_55 = torch.nn.functional.dropout(x_54, 0.0, False, False)
        x_54 = None
        x_56 = x_49 + x_55
        x_49 = x_55 = None
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (768,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        ) = None
        x_gate_3 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_58 = torch._C._nn.linear(
            x_57,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_57 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_3 = torch.nn.functional.silu(x_gate_3, inplace=False)
        x_gate_3 = None
        x_59 = silu_3 * x_58
        silu_3 = x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.nn.functional.layer_norm(
            x_60,
            (2048,),
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_60 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_62 = torch._C._nn.linear(
            x_61,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_61 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_63 = torch.nn.functional.dropout(x_62, 0.0, False, False)
        x_62 = None
        x_64 = x_56 + x_63
        x_56 = x_63 = None
        x_65 = torch.nn.functional.layer_norm(
            x_64,
            (768,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        ) = None
        linear_28 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_24 = linear_28.reshape(1, 197, 12, -1)
        linear_28 = None
        q_8 = reshape_24.transpose(1, 2)
        reshape_24 = None
        linear_29 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_25 = linear_29.reshape(1, 197, 12, -1)
        linear_29 = None
        k_8 = reshape_25.transpose(1, 2)
        reshape_25 = None
        linear_30 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_65 = l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_26 = linear_30.reshape(1, 197, 12, -1)
        linear_30 = None
        v_4 = reshape_26.transpose(1, 2)
        reshape_26 = None
        getitem_52 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_53 = q_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_8 = None
        tensor_split_8 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_8 = tensor_split_8[0]
        cos_emb_8 = tensor_split_8[1]
        tensor_split_8 = None
        mul_20 = getitem_53 * cos_emb_8
        cos_emb_8 = None
        getitem_56 = getitem_53[(Ellipsis, slice(1, None, 2))]
        neg_8 = -getitem_56
        getitem_56 = None
        getitem_57 = getitem_53[(Ellipsis, slice(None, None, 2))]
        getitem_53 = None
        stack_8 = torch.stack([neg_8, getitem_57], -1)
        neg_8 = getitem_57 = None
        reshape_27 = stack_8.reshape((1, 12, 196, 64))
        stack_8 = None
        mul_21 = reshape_27 * sin_emb_8
        reshape_27 = sin_emb_8 = None
        add_17 = mul_20 + mul_21
        mul_20 = mul_21 = None
        cat_9 = torch.cat([getitem_52, add_17], dim=2)
        getitem_52 = add_17 = None
        q_9 = cat_9.type_as(v_4)
        cat_9 = None
        getitem_58 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_59 = k_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_8 = None
        tensor_split_9 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_9 = tensor_split_9[0]
        cos_emb_9 = tensor_split_9[1]
        tensor_split_9 = None
        mul_22 = getitem_59 * cos_emb_9
        cos_emb_9 = None
        getitem_62 = getitem_59[(Ellipsis, slice(1, None, 2))]
        neg_9 = -getitem_62
        getitem_62 = None
        getitem_63 = getitem_59[(Ellipsis, slice(None, None, 2))]
        getitem_59 = None
        stack_9 = torch.stack([neg_9, getitem_63], -1)
        neg_9 = getitem_63 = None
        reshape_28 = stack_9.reshape((1, 12, 196, 64))
        stack_9 = None
        mul_23 = reshape_28 * sin_emb_9
        reshape_28 = sin_emb_9 = None
        add_18 = mul_22 + mul_23
        mul_22 = mul_23 = None
        cat_10 = torch.cat([getitem_58, add_18], dim=2)
        getitem_58 = add_18 = None
        k_9 = cat_10.type_as(v_4)
        cat_10 = None
        x_66 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_4, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_4 = None
        transpose_20 = x_66.transpose(1, 2)
        x_66 = None
        x_67 = transpose_20.reshape(1, 197, 768)
        transpose_20 = None
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (768,),
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_67 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_69 = torch._C._nn.linear(
            x_68,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_68 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_70 = torch.nn.functional.dropout(x_69, 0.0, False, False)
        x_69 = None
        x_71 = x_64 + x_70
        x_64 = x_70 = None
        x_72 = torch.nn.functional.layer_norm(
            x_71,
            (768,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        ) = None
        x_gate_4 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_73 = torch._C._nn.linear(
            x_72,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_72 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_4 = torch.nn.functional.silu(x_gate_4, inplace=False)
        x_gate_4 = None
        x_74 = silu_4 * x_73
        silu_4 = x_73 = None
        x_75 = torch.nn.functional.dropout(x_74, 0.0, False, False)
        x_74 = None
        x_76 = torch.nn.functional.layer_norm(
            x_75,
            (2048,),
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_75 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_77 = torch._C._nn.linear(
            x_76,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_76 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = x_71 + x_78
        x_71 = x_78 = None
        x_80 = torch.nn.functional.layer_norm(
            x_79,
            (768,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        ) = None
        linear_35 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_30 = linear_35.reshape(1, 197, 12, -1)
        linear_35 = None
        q_10 = reshape_30.transpose(1, 2)
        reshape_30 = None
        linear_36 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_31 = linear_36.reshape(1, 197, 12, -1)
        linear_36 = None
        k_10 = reshape_31.transpose(1, 2)
        reshape_31 = None
        linear_37 = torch._C._nn.linear(
            x_80,
            l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_80 = l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_32 = linear_37.reshape(1, 197, 12, -1)
        linear_37 = None
        v_5 = reshape_32.transpose(1, 2)
        reshape_32 = None
        getitem_64 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_65 = q_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_10 = None
        tensor_split_10 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_10 = tensor_split_10[0]
        cos_emb_10 = tensor_split_10[1]
        tensor_split_10 = None
        mul_25 = getitem_65 * cos_emb_10
        cos_emb_10 = None
        getitem_68 = getitem_65[(Ellipsis, slice(1, None, 2))]
        neg_10 = -getitem_68
        getitem_68 = None
        getitem_69 = getitem_65[(Ellipsis, slice(None, None, 2))]
        getitem_65 = None
        stack_10 = torch.stack([neg_10, getitem_69], -1)
        neg_10 = getitem_69 = None
        reshape_33 = stack_10.reshape((1, 12, 196, 64))
        stack_10 = None
        mul_26 = reshape_33 * sin_emb_10
        reshape_33 = sin_emb_10 = None
        add_21 = mul_25 + mul_26
        mul_25 = mul_26 = None
        cat_11 = torch.cat([getitem_64, add_21], dim=2)
        getitem_64 = add_21 = None
        q_11 = cat_11.type_as(v_5)
        cat_11 = None
        getitem_70 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_71 = k_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_10 = None
        tensor_split_11 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_11 = tensor_split_11[0]
        cos_emb_11 = tensor_split_11[1]
        tensor_split_11 = None
        mul_27 = getitem_71 * cos_emb_11
        cos_emb_11 = None
        getitem_74 = getitem_71[(Ellipsis, slice(1, None, 2))]
        neg_11 = -getitem_74
        getitem_74 = None
        getitem_75 = getitem_71[(Ellipsis, slice(None, None, 2))]
        getitem_71 = None
        stack_11 = torch.stack([neg_11, getitem_75], -1)
        neg_11 = getitem_75 = None
        reshape_34 = stack_11.reshape((1, 12, 196, 64))
        stack_11 = None
        mul_28 = reshape_34 * sin_emb_11
        reshape_34 = sin_emb_11 = None
        add_22 = mul_27 + mul_28
        mul_27 = mul_28 = None
        cat_12 = torch.cat([getitem_70, add_22], dim=2)
        getitem_70 = add_22 = None
        k_11 = cat_12.type_as(v_5)
        cat_12 = None
        x_81 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_5, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_5 = None
        transpose_24 = x_81.transpose(1, 2)
        x_81 = None
        x_82 = transpose_24.reshape(1, 197, 768)
        transpose_24 = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (768,),
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_82 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_84 = torch._C._nn.linear(
            x_83,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_83 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_85 = torch.nn.functional.dropout(x_84, 0.0, False, False)
        x_84 = None
        x_86 = x_79 + x_85
        x_79 = x_85 = None
        x_87 = torch.nn.functional.layer_norm(
            x_86,
            (768,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        ) = None
        x_gate_5 = torch._C._nn.linear(
            x_87,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_88 = torch._C._nn.linear(
            x_87,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_87 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_5 = torch.nn.functional.silu(x_gate_5, inplace=False)
        x_gate_5 = None
        x_89 = silu_5 * x_88
        silu_5 = x_88 = None
        x_90 = torch.nn.functional.dropout(x_89, 0.0, False, False)
        x_89 = None
        x_91 = torch.nn.functional.layer_norm(
            x_90,
            (2048,),
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_90 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = x_86 + x_93
        x_86 = x_93 = None
        x_95 = torch.nn.functional.layer_norm(
            x_94,
            (768,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        ) = None
        linear_42 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_36 = linear_42.reshape(1, 197, 12, -1)
        linear_42 = None
        q_12 = reshape_36.transpose(1, 2)
        reshape_36 = None
        linear_43 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_37 = linear_43.reshape(1, 197, 12, -1)
        linear_43 = None
        k_12 = reshape_37.transpose(1, 2)
        reshape_37 = None
        linear_44 = torch._C._nn.linear(
            x_95,
            l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_95 = l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_38 = linear_44.reshape(1, 197, 12, -1)
        linear_44 = None
        v_6 = reshape_38.transpose(1, 2)
        reshape_38 = None
        getitem_76 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_77 = q_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_12 = None
        tensor_split_12 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_12 = tensor_split_12[0]
        cos_emb_12 = tensor_split_12[1]
        tensor_split_12 = None
        mul_30 = getitem_77 * cos_emb_12
        cos_emb_12 = None
        getitem_80 = getitem_77[(Ellipsis, slice(1, None, 2))]
        neg_12 = -getitem_80
        getitem_80 = None
        getitem_81 = getitem_77[(Ellipsis, slice(None, None, 2))]
        getitem_77 = None
        stack_12 = torch.stack([neg_12, getitem_81], -1)
        neg_12 = getitem_81 = None
        reshape_39 = stack_12.reshape((1, 12, 196, 64))
        stack_12 = None
        mul_31 = reshape_39 * sin_emb_12
        reshape_39 = sin_emb_12 = None
        add_25 = mul_30 + mul_31
        mul_30 = mul_31 = None
        cat_13 = torch.cat([getitem_76, add_25], dim=2)
        getitem_76 = add_25 = None
        q_13 = cat_13.type_as(v_6)
        cat_13 = None
        getitem_82 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_83 = k_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_12 = None
        tensor_split_13 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_13 = tensor_split_13[0]
        cos_emb_13 = tensor_split_13[1]
        tensor_split_13 = None
        mul_32 = getitem_83 * cos_emb_13
        cos_emb_13 = None
        getitem_86 = getitem_83[(Ellipsis, slice(1, None, 2))]
        neg_13 = -getitem_86
        getitem_86 = None
        getitem_87 = getitem_83[(Ellipsis, slice(None, None, 2))]
        getitem_83 = None
        stack_13 = torch.stack([neg_13, getitem_87], -1)
        neg_13 = getitem_87 = None
        reshape_40 = stack_13.reshape((1, 12, 196, 64))
        stack_13 = None
        mul_33 = reshape_40 * sin_emb_13
        reshape_40 = sin_emb_13 = None
        add_26 = mul_32 + mul_33
        mul_32 = mul_33 = None
        cat_14 = torch.cat([getitem_82, add_26], dim=2)
        getitem_82 = add_26 = None
        k_13 = cat_14.type_as(v_6)
        cat_14 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_6, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_6 = None
        transpose_28 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_28.reshape(1, 197, 768)
        transpose_28 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (768,),
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_97 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_98 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.dropout(x_99, 0.0, False, False)
        x_99 = None
        x_101 = x_94 + x_100
        x_94 = x_100 = None
        x_102 = torch.nn.functional.layer_norm(
            x_101,
            (768,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        ) = None
        x_gate_6 = torch._C._nn.linear(
            x_102,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_103 = torch._C._nn.linear(
            x_102,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_102 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_6 = torch.nn.functional.silu(x_gate_6, inplace=False)
        x_gate_6 = None
        x_104 = silu_6 * x_103
        silu_6 = x_103 = None
        x_105 = torch.nn.functional.dropout(x_104, 0.0, False, False)
        x_104 = None
        x_106 = torch.nn.functional.layer_norm(
            x_105,
            (2048,),
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_105 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_107 = torch._C._nn.linear(
            x_106,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_106 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_108 = torch.nn.functional.dropout(x_107, 0.0, False, False)
        x_107 = None
        x_109 = x_101 + x_108
        x_101 = x_108 = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (768,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        ) = None
        linear_49 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_42 = linear_49.reshape(1, 197, 12, -1)
        linear_49 = None
        q_14 = reshape_42.transpose(1, 2)
        reshape_42 = None
        linear_50 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_43 = linear_50.reshape(1, 197, 12, -1)
        linear_50 = None
        k_14 = reshape_43.transpose(1, 2)
        reshape_43 = None
        linear_51 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_110 = l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_44 = linear_51.reshape(1, 197, 12, -1)
        linear_51 = None
        v_7 = reshape_44.transpose(1, 2)
        reshape_44 = None
        getitem_88 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_89 = q_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_14 = None
        tensor_split_14 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_14 = tensor_split_14[0]
        cos_emb_14 = tensor_split_14[1]
        tensor_split_14 = None
        mul_35 = getitem_89 * cos_emb_14
        cos_emb_14 = None
        getitem_92 = getitem_89[(Ellipsis, slice(1, None, 2))]
        neg_14 = -getitem_92
        getitem_92 = None
        getitem_93 = getitem_89[(Ellipsis, slice(None, None, 2))]
        getitem_89 = None
        stack_14 = torch.stack([neg_14, getitem_93], -1)
        neg_14 = getitem_93 = None
        reshape_45 = stack_14.reshape((1, 12, 196, 64))
        stack_14 = None
        mul_36 = reshape_45 * sin_emb_14
        reshape_45 = sin_emb_14 = None
        add_29 = mul_35 + mul_36
        mul_35 = mul_36 = None
        cat_15 = torch.cat([getitem_88, add_29], dim=2)
        getitem_88 = add_29 = None
        q_15 = cat_15.type_as(v_7)
        cat_15 = None
        getitem_94 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_95 = k_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_14 = None
        tensor_split_15 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_15 = tensor_split_15[0]
        cos_emb_15 = tensor_split_15[1]
        tensor_split_15 = None
        mul_37 = getitem_95 * cos_emb_15
        cos_emb_15 = None
        getitem_98 = getitem_95[(Ellipsis, slice(1, None, 2))]
        neg_15 = -getitem_98
        getitem_98 = None
        getitem_99 = getitem_95[(Ellipsis, slice(None, None, 2))]
        getitem_95 = None
        stack_15 = torch.stack([neg_15, getitem_99], -1)
        neg_15 = getitem_99 = None
        reshape_46 = stack_15.reshape((1, 12, 196, 64))
        stack_15 = None
        mul_38 = reshape_46 * sin_emb_15
        reshape_46 = sin_emb_15 = None
        add_30 = mul_37 + mul_38
        mul_37 = mul_38 = None
        cat_16 = torch.cat([getitem_94, add_30], dim=2)
        getitem_94 = add_30 = None
        k_15 = cat_16.type_as(v_7)
        cat_16 = None
        x_111 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_7, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_7 = None
        transpose_32 = x_111.transpose(1, 2)
        x_111 = None
        x_112 = transpose_32.reshape(1, 197, 768)
        transpose_32 = None
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (768,),
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_112 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_113 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.dropout(x_114, 0.0, False, False)
        x_114 = None
        x_116 = x_109 + x_115
        x_109 = x_115 = None
        x_117 = torch.nn.functional.layer_norm(
            x_116,
            (768,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        ) = None
        x_gate_7 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_7 = torch.nn.functional.silu(x_gate_7, inplace=False)
        x_gate_7 = None
        x_119 = silu_7 * x_118
        silu_7 = x_118 = None
        x_120 = torch.nn.functional.dropout(x_119, 0.0, False, False)
        x_119 = None
        x_121 = torch.nn.functional.layer_norm(
            x_120,
            (2048,),
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_120 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_122 = torch._C._nn.linear(
            x_121,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_121 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.dropout(x_122, 0.0, False, False)
        x_122 = None
        x_124 = x_116 + x_123
        x_116 = x_123 = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (768,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        ) = None
        linear_56 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_48 = linear_56.reshape(1, 197, 12, -1)
        linear_56 = None
        q_16 = reshape_48.transpose(1, 2)
        reshape_48 = None
        linear_57 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_49 = linear_57.reshape(1, 197, 12, -1)
        linear_57 = None
        k_16 = reshape_49.transpose(1, 2)
        reshape_49 = None
        linear_58 = torch._C._nn.linear(
            x_125,
            l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_125 = l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_50 = linear_58.reshape(1, 197, 12, -1)
        linear_58 = None
        v_8 = reshape_50.transpose(1, 2)
        reshape_50 = None
        getitem_100 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_101 = q_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_16 = None
        tensor_split_16 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_16 = tensor_split_16[0]
        cos_emb_16 = tensor_split_16[1]
        tensor_split_16 = None
        mul_40 = getitem_101 * cos_emb_16
        cos_emb_16 = None
        getitem_104 = getitem_101[(Ellipsis, slice(1, None, 2))]
        neg_16 = -getitem_104
        getitem_104 = None
        getitem_105 = getitem_101[(Ellipsis, slice(None, None, 2))]
        getitem_101 = None
        stack_16 = torch.stack([neg_16, getitem_105], -1)
        neg_16 = getitem_105 = None
        reshape_51 = stack_16.reshape((1, 12, 196, 64))
        stack_16 = None
        mul_41 = reshape_51 * sin_emb_16
        reshape_51 = sin_emb_16 = None
        add_33 = mul_40 + mul_41
        mul_40 = mul_41 = None
        cat_17 = torch.cat([getitem_100, add_33], dim=2)
        getitem_100 = add_33 = None
        q_17 = cat_17.type_as(v_8)
        cat_17 = None
        getitem_106 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_107 = k_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_16 = None
        tensor_split_17 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_17 = tensor_split_17[0]
        cos_emb_17 = tensor_split_17[1]
        tensor_split_17 = None
        mul_42 = getitem_107 * cos_emb_17
        cos_emb_17 = None
        getitem_110 = getitem_107[(Ellipsis, slice(1, None, 2))]
        neg_17 = -getitem_110
        getitem_110 = None
        getitem_111 = getitem_107[(Ellipsis, slice(None, None, 2))]
        getitem_107 = None
        stack_17 = torch.stack([neg_17, getitem_111], -1)
        neg_17 = getitem_111 = None
        reshape_52 = stack_17.reshape((1, 12, 196, 64))
        stack_17 = None
        mul_43 = reshape_52 * sin_emb_17
        reshape_52 = sin_emb_17 = None
        add_34 = mul_42 + mul_43
        mul_42 = mul_43 = None
        cat_18 = torch.cat([getitem_106, add_34], dim=2)
        getitem_106 = add_34 = None
        k_17 = cat_18.type_as(v_8)
        cat_18 = None
        x_126 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_8, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_8 = None
        transpose_36 = x_126.transpose(1, 2)
        x_126 = None
        x_127 = transpose_36.reshape(1, 197, 768)
        transpose_36 = None
        x_128 = torch.nn.functional.layer_norm(
            x_127,
            (768,),
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_127 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_128 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = x_124 + x_130
        x_124 = x_130 = None
        x_132 = torch.nn.functional.layer_norm(
            x_131,
            (768,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        ) = None
        x_gate_8 = torch._C._nn.linear(
            x_132,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_133 = torch._C._nn.linear(
            x_132,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_132 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_8 = torch.nn.functional.silu(x_gate_8, inplace=False)
        x_gate_8 = None
        x_134 = silu_8 * x_133
        silu_8 = x_133 = None
        x_135 = torch.nn.functional.dropout(x_134, 0.0, False, False)
        x_134 = None
        x_136 = torch.nn.functional.layer_norm(
            x_135,
            (2048,),
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_135 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_136 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = x_131 + x_138
        x_131 = x_138 = None
        x_140 = torch.nn.functional.layer_norm(
            x_139,
            (768,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        ) = None
        linear_63 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_
        ) = None
        reshape_54 = linear_63.reshape(1, 197, 12, -1)
        linear_63 = None
        q_18 = reshape_54.transpose(1, 2)
        reshape_54 = None
        linear_64 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_55 = linear_64.reshape(1, 197, 12, -1)
        linear_64 = None
        k_18 = reshape_55.transpose(1, 2)
        reshape_55 = None
        linear_65 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_140 = l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_
        ) = None
        reshape_56 = linear_65.reshape(1, 197, 12, -1)
        linear_65 = None
        v_9 = reshape_56.transpose(1, 2)
        reshape_56 = None
        getitem_112 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_113 = q_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_18 = None
        tensor_split_18 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_18 = tensor_split_18[0]
        cos_emb_18 = tensor_split_18[1]
        tensor_split_18 = None
        mul_45 = getitem_113 * cos_emb_18
        cos_emb_18 = None
        getitem_116 = getitem_113[(Ellipsis, slice(1, None, 2))]
        neg_18 = -getitem_116
        getitem_116 = None
        getitem_117 = getitem_113[(Ellipsis, slice(None, None, 2))]
        getitem_113 = None
        stack_18 = torch.stack([neg_18, getitem_117], -1)
        neg_18 = getitem_117 = None
        reshape_57 = stack_18.reshape((1, 12, 196, 64))
        stack_18 = None
        mul_46 = reshape_57 * sin_emb_18
        reshape_57 = sin_emb_18 = None
        add_37 = mul_45 + mul_46
        mul_45 = mul_46 = None
        cat_19 = torch.cat([getitem_112, add_37], dim=2)
        getitem_112 = add_37 = None
        q_19 = cat_19.type_as(v_9)
        cat_19 = None
        getitem_118 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_119 = k_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_18 = None
        tensor_split_19 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_19 = tensor_split_19[0]
        cos_emb_19 = tensor_split_19[1]
        tensor_split_19 = None
        mul_47 = getitem_119 * cos_emb_19
        cos_emb_19 = None
        getitem_122 = getitem_119[(Ellipsis, slice(1, None, 2))]
        neg_19 = -getitem_122
        getitem_122 = None
        getitem_123 = getitem_119[(Ellipsis, slice(None, None, 2))]
        getitem_119 = None
        stack_19 = torch.stack([neg_19, getitem_123], -1)
        neg_19 = getitem_123 = None
        reshape_58 = stack_19.reshape((1, 12, 196, 64))
        stack_19 = None
        mul_48 = reshape_58 * sin_emb_19
        reshape_58 = sin_emb_19 = None
        add_38 = mul_47 + mul_48
        mul_47 = mul_48 = None
        cat_20 = torch.cat([getitem_118, add_38], dim=2)
        getitem_118 = add_38 = None
        k_19 = cat_20.type_as(v_9)
        cat_20 = None
        x_141 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_9, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_9 = None
        transpose_40 = x_141.transpose(1, 2)
        x_141 = None
        x_142 = transpose_40.reshape(1, 197, 768)
        transpose_40 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (768,),
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_142 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_143 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = x_139 + x_145
        x_139 = x_145 = None
        x_147 = torch.nn.functional.layer_norm(
            x_146,
            (768,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        ) = None
        x_gate_9 = torch._C._nn.linear(
            x_147,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_147 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_9 = torch.nn.functional.silu(x_gate_9, inplace=False)
        x_gate_9 = None
        x_149 = silu_9 * x_148
        silu_9 = x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.0, False, False)
        x_149 = None
        x_151 = torch.nn.functional.layer_norm(
            x_150,
            (2048,),
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_150 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_151 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.dropout(x_152, 0.0, False, False)
        x_152 = None
        x_154 = x_146 + x_153
        x_146 = x_153 = None
        x_155 = torch.nn.functional.layer_norm(
            x_154,
            (768,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        ) = None
        linear_70 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_60 = linear_70.reshape(1, 197, 12, -1)
        linear_70 = None
        q_20 = reshape_60.transpose(1, 2)
        reshape_60 = None
        linear_71 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_61 = linear_71.reshape(1, 197, 12, -1)
        linear_71 = None
        k_20 = reshape_61.transpose(1, 2)
        reshape_61 = None
        linear_72 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_155 = l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_62 = linear_72.reshape(1, 197, 12, -1)
        linear_72 = None
        v_10 = reshape_62.transpose(1, 2)
        reshape_62 = None
        getitem_124 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_125 = q_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_20 = None
        tensor_split_20 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_20 = tensor_split_20[0]
        cos_emb_20 = tensor_split_20[1]
        tensor_split_20 = None
        mul_50 = getitem_125 * cos_emb_20
        cos_emb_20 = None
        getitem_128 = getitem_125[(Ellipsis, slice(1, None, 2))]
        neg_20 = -getitem_128
        getitem_128 = None
        getitem_129 = getitem_125[(Ellipsis, slice(None, None, 2))]
        getitem_125 = None
        stack_20 = torch.stack([neg_20, getitem_129], -1)
        neg_20 = getitem_129 = None
        reshape_63 = stack_20.reshape((1, 12, 196, 64))
        stack_20 = None
        mul_51 = reshape_63 * sin_emb_20
        reshape_63 = sin_emb_20 = None
        add_41 = mul_50 + mul_51
        mul_50 = mul_51 = None
        cat_21 = torch.cat([getitem_124, add_41], dim=2)
        getitem_124 = add_41 = None
        q_21 = cat_21.type_as(v_10)
        cat_21 = None
        getitem_130 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_131 = k_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_20 = None
        tensor_split_21 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_21 = tensor_split_21[0]
        cos_emb_21 = tensor_split_21[1]
        tensor_split_21 = None
        mul_52 = getitem_131 * cos_emb_21
        cos_emb_21 = None
        getitem_134 = getitem_131[(Ellipsis, slice(1, None, 2))]
        neg_21 = -getitem_134
        getitem_134 = None
        getitem_135 = getitem_131[(Ellipsis, slice(None, None, 2))]
        getitem_131 = None
        stack_21 = torch.stack([neg_21, getitem_135], -1)
        neg_21 = getitem_135 = None
        reshape_64 = stack_21.reshape((1, 12, 196, 64))
        stack_21 = None
        mul_53 = reshape_64 * sin_emb_21
        reshape_64 = sin_emb_21 = None
        add_42 = mul_52 + mul_53
        mul_52 = mul_53 = None
        cat_22 = torch.cat([getitem_130, add_42], dim=2)
        getitem_130 = add_42 = None
        k_21 = cat_22.type_as(v_10)
        cat_22 = None
        x_156 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_10, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_10 = None
        transpose_44 = x_156.transpose(1, 2)
        x_156 = None
        x_157 = transpose_44.reshape(1, 197, 768)
        transpose_44 = None
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (768,),
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_157 = l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_159 = torch._C._nn.linear(
            x_158,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_158 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_160 = torch.nn.functional.dropout(x_159, 0.0, False, False)
        x_159 = None
        x_161 = x_154 + x_160
        x_154 = x_160 = None
        x_162 = torch.nn.functional.layer_norm(
            x_161,
            (768,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        ) = None
        x_gate_10 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_162 = l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_10 = torch.nn.functional.silu(x_gate_10, inplace=False)
        x_gate_10 = None
        x_164 = silu_10 * x_163
        silu_10 = x_163 = None
        x_165 = torch.nn.functional.dropout(x_164, 0.0, False, False)
        x_164 = None
        x_166 = torch.nn.functional.layer_norm(
            x_165,
            (2048,),
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_165 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_166 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_168 = torch.nn.functional.dropout(x_167, 0.0, False, False)
        x_167 = None
        x_169 = x_161 + x_168
        x_161 = x_168 = None
        x_170 = torch.nn.functional.layer_norm(
            x_169,
            (768,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        ) = None
        linear_77 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_66 = linear_77.reshape(1, 197, 12, -1)
        linear_77 = None
        q_22 = reshape_66.transpose(1, 2)
        reshape_66 = None
        linear_78 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_67 = linear_78.reshape(1, 197, 12, -1)
        linear_78 = None
        k_22 = reshape_67.transpose(1, 2)
        reshape_67 = None
        linear_79 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_170 = l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_68 = linear_79.reshape(1, 197, 12, -1)
        linear_79 = None
        v_11 = reshape_68.transpose(1, 2)
        reshape_68 = None
        getitem_136 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_137 = q_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_22 = None
        tensor_split_22 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_22 = tensor_split_22[0]
        cos_emb_22 = tensor_split_22[1]
        tensor_split_22 = None
        mul_55 = getitem_137 * cos_emb_22
        cos_emb_22 = None
        getitem_140 = getitem_137[(Ellipsis, slice(1, None, 2))]
        neg_22 = -getitem_140
        getitem_140 = None
        getitem_141 = getitem_137[(Ellipsis, slice(None, None, 2))]
        getitem_137 = None
        stack_22 = torch.stack([neg_22, getitem_141], -1)
        neg_22 = getitem_141 = None
        reshape_69 = stack_22.reshape((1, 12, 196, 64))
        stack_22 = None
        mul_56 = reshape_69 * sin_emb_22
        reshape_69 = sin_emb_22 = None
        add_45 = mul_55 + mul_56
        mul_55 = mul_56 = None
        cat_23 = torch.cat([getitem_136, add_45], dim=2)
        getitem_136 = add_45 = None
        q_23 = cat_23.type_as(v_11)
        cat_23 = None
        getitem_142 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_143 = k_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_22 = None
        tensor_split_23 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        l_self_modules_rope_buffers_pos_embed_ = None
        sin_emb_23 = tensor_split_23[0]
        cos_emb_23 = tensor_split_23[1]
        tensor_split_23 = None
        mul_57 = getitem_143 * cos_emb_23
        cos_emb_23 = None
        getitem_146 = getitem_143[(Ellipsis, slice(1, None, 2))]
        neg_23 = -getitem_146
        getitem_146 = None
        getitem_147 = getitem_143[(Ellipsis, slice(None, None, 2))]
        getitem_143 = None
        stack_23 = torch.stack([neg_23, getitem_147], -1)
        neg_23 = getitem_147 = None
        reshape_70 = stack_23.reshape((1, 12, 196, 64))
        stack_23 = None
        mul_58 = reshape_70 * sin_emb_23
        reshape_70 = sin_emb_23 = None
        add_46 = mul_57 + mul_58
        mul_57 = mul_58 = None
        cat_24 = torch.cat([getitem_142, add_46], dim=2)
        getitem_142 = add_46 = None
        k_23 = cat_24.type_as(v_11)
        cat_24 = None
        x_171 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_11, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_11 = None
        transpose_48 = x_171.transpose(1, 2)
        x_171 = None
        x_172 = transpose_48.reshape(1, 197, 768)
        transpose_48 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (768,),
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_172 = l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_173 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.dropout(x_174, 0.0, False, False)
        x_174 = None
        x_176 = x_169 + x_175
        x_169 = x_175 = None
        x_177 = torch.nn.functional.layer_norm(
            x_176,
            (768,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        ) = None
        x_gate_11 = torch._C._nn.linear(
            x_177,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_178 = torch._C._nn.linear(
            x_177,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_177 = l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_11 = torch.nn.functional.silu(x_gate_11, inplace=False)
        x_gate_11 = None
        x_179 = silu_11 * x_178
        silu_11 = x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.0, False, False)
        x_179 = None
        x_181 = torch.nn.functional.layer_norm(
            x_180,
            (2048,),
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_180 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_181 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_183 = torch.nn.functional.dropout(x_182, 0.0, False, False)
        x_182 = None
        x_184 = x_176 + x_183
        x_176 = x_183 = None
        x_185 = torch.nn.functional.layer_norm(
            x_184,
            (768,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_184 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_186 = x_185[(slice(None, None, None), 0)]
        x_185 = None
        x_187 = torch.nn.functional.dropout(x_186, 0.0, False, False)
        x_186 = None
        x_188 = torch._C._nn.linear(
            x_187,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_187 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_188,)
