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
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_12_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_13_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_14_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_15_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_16_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_17_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_18_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_19_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_20_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_21_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_22_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_bias_ = L_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_bias_
        l_self_modules_blocks_modules_23_modules_attn_modules_k_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_k_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_bias_ = L_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_bias_
        l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_weight_ = L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_weight_
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_patch_embed_modules_proj_parameters_weight_,
            l_self_modules_patch_embed_modules_proj_parameters_bias_,
            (14, 14),
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
            (1024,),
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
        reshape = linear.reshape(1, 257, 16, -1)
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
        reshape_1 = linear_1.reshape(1, 257, 16, -1)
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
        reshape_2 = linear_2.reshape(1, 257, 16, -1)
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
        reshape_3 = stack.reshape((1, 16, 256, 64))
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
        reshape_4 = stack_1.reshape((1, 16, 256, 64))
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
        x_7 = transpose_4.reshape(1, 257, 1024)
        transpose_4 = None
        x_8 = torch.nn.functional.layer_norm(
            x_7,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_6 = linear_7.reshape(1, 257, 16, -1)
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
        reshape_7 = linear_8.reshape(1, 257, 16, -1)
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
        reshape_8 = linear_9.reshape(1, 257, 16, -1)
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
        reshape_9 = stack_2.reshape((1, 16, 256, 64))
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
        reshape_10 = stack_3.reshape((1, 16, 256, 64))
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
        x_22 = transpose_8.reshape(1, 257, 1024)
        transpose_8 = None
        x_23 = torch.nn.functional.layer_norm(
            x_22,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_12 = linear_14.reshape(1, 257, 16, -1)
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
        reshape_13 = linear_15.reshape(1, 257, 16, -1)
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
        reshape_14 = linear_16.reshape(1, 257, 16, -1)
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
        reshape_15 = stack_4.reshape((1, 16, 256, 64))
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
        reshape_16 = stack_5.reshape((1, 16, 256, 64))
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
        x_37 = transpose_12.reshape(1, 257, 1024)
        transpose_12 = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_18 = linear_21.reshape(1, 257, 16, -1)
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
        reshape_19 = linear_22.reshape(1, 257, 16, -1)
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
        reshape_20 = linear_23.reshape(1, 257, 16, -1)
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
        reshape_21 = stack_6.reshape((1, 16, 256, 64))
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
        reshape_22 = stack_7.reshape((1, 16, 256, 64))
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
        x_52 = transpose_16.reshape(1, 257, 1024)
        transpose_16 = None
        x_53 = torch.nn.functional.layer_norm(
            x_52,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_24 = linear_28.reshape(1, 257, 16, -1)
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
        reshape_25 = linear_29.reshape(1, 257, 16, -1)
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
        reshape_26 = linear_30.reshape(1, 257, 16, -1)
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
        reshape_27 = stack_8.reshape((1, 16, 256, 64))
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
        reshape_28 = stack_9.reshape((1, 16, 256, 64))
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
        x_67 = transpose_20.reshape(1, 257, 1024)
        transpose_20 = None
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_30 = linear_35.reshape(1, 257, 16, -1)
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
        reshape_31 = linear_36.reshape(1, 257, 16, -1)
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
        reshape_32 = linear_37.reshape(1, 257, 16, -1)
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
        reshape_33 = stack_10.reshape((1, 16, 256, 64))
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
        reshape_34 = stack_11.reshape((1, 16, 256, 64))
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
        x_82 = transpose_24.reshape(1, 257, 1024)
        transpose_24 = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_36 = linear_42.reshape(1, 257, 16, -1)
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
        reshape_37 = linear_43.reshape(1, 257, 16, -1)
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
        reshape_38 = linear_44.reshape(1, 257, 16, -1)
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
        reshape_39 = stack_12.reshape((1, 16, 256, 64))
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
        reshape_40 = stack_13.reshape((1, 16, 256, 64))
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
        x_97 = transpose_28.reshape(1, 257, 1024)
        transpose_28 = None
        x_98 = torch.nn.functional.layer_norm(
            x_97,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_42 = linear_49.reshape(1, 257, 16, -1)
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
        reshape_43 = linear_50.reshape(1, 257, 16, -1)
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
        reshape_44 = linear_51.reshape(1, 257, 16, -1)
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
        reshape_45 = stack_14.reshape((1, 16, 256, 64))
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
        reshape_46 = stack_15.reshape((1, 16, 256, 64))
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
        x_112 = transpose_32.reshape(1, 257, 1024)
        transpose_32 = None
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_48 = linear_56.reshape(1, 257, 16, -1)
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
        reshape_49 = linear_57.reshape(1, 257, 16, -1)
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
        reshape_50 = linear_58.reshape(1, 257, 16, -1)
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
        reshape_51 = stack_16.reshape((1, 16, 256, 64))
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
        reshape_52 = stack_17.reshape((1, 16, 256, 64))
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
        x_127 = transpose_36.reshape(1, 257, 1024)
        transpose_36 = None
        x_128 = torch.nn.functional.layer_norm(
            x_127,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_54 = linear_63.reshape(1, 257, 16, -1)
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
        reshape_55 = linear_64.reshape(1, 257, 16, -1)
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
        reshape_56 = linear_65.reshape(1, 257, 16, -1)
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
        reshape_57 = stack_18.reshape((1, 16, 256, 64))
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
        reshape_58 = stack_19.reshape((1, 16, 256, 64))
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
        x_142 = transpose_40.reshape(1, 257, 1024)
        transpose_40 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_60 = linear_70.reshape(1, 257, 16, -1)
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
        reshape_61 = linear_71.reshape(1, 257, 16, -1)
        linear_71 = None
        k_20 = reshape_61.transpose(1, 2)
        reshape_61 = None
        linear_72 = torch._C._nn.linear(
            x_155,
            l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_155 = l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_62 = linear_72.reshape(1, 257, 16, -1)
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
        reshape_63 = stack_20.reshape((1, 16, 256, 64))
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
        reshape_64 = stack_21.reshape((1, 16, 256, 64))
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
        x_157 = transpose_44.reshape(1, 257, 1024)
        transpose_44 = None
        x_158 = torch.nn.functional.layer_norm(
            x_157,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
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
        reshape_66 = linear_77.reshape(1, 257, 16, -1)
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
        reshape_67 = linear_78.reshape(1, 257, 16, -1)
        linear_78 = None
        k_22 = reshape_67.transpose(1, 2)
        reshape_67 = None
        linear_79 = torch._C._nn.linear(
            x_170,
            l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_170 = l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_68 = linear_79.reshape(1, 257, 16, -1)
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
        reshape_69 = stack_22.reshape((1, 16, 256, 64))
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
        reshape_70 = stack_23.reshape((1, 16, 256, 64))
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
        x_172 = transpose_48.reshape(1, 257, 1024)
        transpose_48 = None
        x_173 = torch.nn.functional.layer_norm(
            x_172,
            (1024,),
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
            (1024,),
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
            (2730,),
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
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_185,
            l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_12_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_72 = linear_84.reshape(1, 257, 16, -1)
        linear_84 = None
        q_24 = reshape_72.transpose(1, 2)
        reshape_72 = None
        linear_85 = torch._C._nn.linear(
            x_185,
            l_self_modules_blocks_modules_12_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_73 = linear_85.reshape(1, 257, 16, -1)
        linear_85 = None
        k_24 = reshape_73.transpose(1, 2)
        reshape_73 = None
        linear_86 = torch._C._nn.linear(
            x_185,
            l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_185 = l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_12_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_74 = linear_86.reshape(1, 257, 16, -1)
        linear_86 = None
        v_12 = reshape_74.transpose(1, 2)
        reshape_74 = None
        getitem_148 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_149 = q_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_24 = None
        tensor_split_24 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_24 = tensor_split_24[0]
        cos_emb_24 = tensor_split_24[1]
        tensor_split_24 = None
        mul_60 = getitem_149 * cos_emb_24
        cos_emb_24 = None
        getitem_152 = getitem_149[(Ellipsis, slice(1, None, 2))]
        neg_24 = -getitem_152
        getitem_152 = None
        getitem_153 = getitem_149[(Ellipsis, slice(None, None, 2))]
        getitem_149 = None
        stack_24 = torch.stack([neg_24, getitem_153], -1)
        neg_24 = getitem_153 = None
        reshape_75 = stack_24.reshape((1, 16, 256, 64))
        stack_24 = None
        mul_61 = reshape_75 * sin_emb_24
        reshape_75 = sin_emb_24 = None
        add_49 = mul_60 + mul_61
        mul_60 = mul_61 = None
        cat_25 = torch.cat([getitem_148, add_49], dim=2)
        getitem_148 = add_49 = None
        q_25 = cat_25.type_as(v_12)
        cat_25 = None
        getitem_154 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_155 = k_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_24 = None
        tensor_split_25 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_25 = tensor_split_25[0]
        cos_emb_25 = tensor_split_25[1]
        tensor_split_25 = None
        mul_62 = getitem_155 * cos_emb_25
        cos_emb_25 = None
        getitem_158 = getitem_155[(Ellipsis, slice(1, None, 2))]
        neg_25 = -getitem_158
        getitem_158 = None
        getitem_159 = getitem_155[(Ellipsis, slice(None, None, 2))]
        getitem_155 = None
        stack_25 = torch.stack([neg_25, getitem_159], -1)
        neg_25 = getitem_159 = None
        reshape_76 = stack_25.reshape((1, 16, 256, 64))
        stack_25 = None
        mul_63 = reshape_76 * sin_emb_25
        reshape_76 = sin_emb_25 = None
        add_50 = mul_62 + mul_63
        mul_62 = mul_63 = None
        cat_26 = torch.cat([getitem_154, add_50], dim=2)
        getitem_154 = add_50 = None
        k_25 = cat_26.type_as(v_12)
        cat_26 = None
        x_186 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_12, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_12 = None
        transpose_52 = x_186.transpose(1, 2)
        x_186 = None
        x_187 = transpose_52.reshape(1, 257, 1024)
        transpose_52 = None
        x_188 = torch.nn.functional.layer_norm(
            x_187,
            (1024,),
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_187 = l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_188 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = x_184 + x_190
        x_184 = x_190 = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (1024,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        ) = None
        x_gate_12 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_192 = l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_12 = torch.nn.functional.silu(x_gate_12, inplace=False)
        x_gate_12 = None
        x_194 = silu_12 * x_193
        silu_12 = x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch.nn.functional.layer_norm(
            x_195,
            (2730,),
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_195 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_197 = torch._C._nn.linear(
            x_196,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_196 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_198 = torch.nn.functional.dropout(x_197, 0.0, False, False)
        x_197 = None
        x_199 = x_191 + x_198
        x_191 = x_198 = None
        x_200 = torch.nn.functional.layer_norm(
            x_199,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        ) = None
        linear_91 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_13_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_78 = linear_91.reshape(1, 257, 16, -1)
        linear_91 = None
        q_26 = reshape_78.transpose(1, 2)
        reshape_78 = None
        linear_92 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_13_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_79 = linear_92.reshape(1, 257, 16, -1)
        linear_92 = None
        k_26 = reshape_79.transpose(1, 2)
        reshape_79 = None
        linear_93 = torch._C._nn.linear(
            x_200,
            l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_200 = l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_13_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_80 = linear_93.reshape(1, 257, 16, -1)
        linear_93 = None
        v_13 = reshape_80.transpose(1, 2)
        reshape_80 = None
        getitem_160 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_161 = q_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_26 = None
        tensor_split_26 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_26 = tensor_split_26[0]
        cos_emb_26 = tensor_split_26[1]
        tensor_split_26 = None
        mul_65 = getitem_161 * cos_emb_26
        cos_emb_26 = None
        getitem_164 = getitem_161[(Ellipsis, slice(1, None, 2))]
        neg_26 = -getitem_164
        getitem_164 = None
        getitem_165 = getitem_161[(Ellipsis, slice(None, None, 2))]
        getitem_161 = None
        stack_26 = torch.stack([neg_26, getitem_165], -1)
        neg_26 = getitem_165 = None
        reshape_81 = stack_26.reshape((1, 16, 256, 64))
        stack_26 = None
        mul_66 = reshape_81 * sin_emb_26
        reshape_81 = sin_emb_26 = None
        add_53 = mul_65 + mul_66
        mul_65 = mul_66 = None
        cat_27 = torch.cat([getitem_160, add_53], dim=2)
        getitem_160 = add_53 = None
        q_27 = cat_27.type_as(v_13)
        cat_27 = None
        getitem_166 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_167 = k_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_26 = None
        tensor_split_27 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_27 = tensor_split_27[0]
        cos_emb_27 = tensor_split_27[1]
        tensor_split_27 = None
        mul_67 = getitem_167 * cos_emb_27
        cos_emb_27 = None
        getitem_170 = getitem_167[(Ellipsis, slice(1, None, 2))]
        neg_27 = -getitem_170
        getitem_170 = None
        getitem_171 = getitem_167[(Ellipsis, slice(None, None, 2))]
        getitem_167 = None
        stack_27 = torch.stack([neg_27, getitem_171], -1)
        neg_27 = getitem_171 = None
        reshape_82 = stack_27.reshape((1, 16, 256, 64))
        stack_27 = None
        mul_68 = reshape_82 * sin_emb_27
        reshape_82 = sin_emb_27 = None
        add_54 = mul_67 + mul_68
        mul_67 = mul_68 = None
        cat_28 = torch.cat([getitem_166, add_54], dim=2)
        getitem_166 = add_54 = None
        k_27 = cat_28.type_as(v_13)
        cat_28 = None
        x_201 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_13, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_13 = None
        transpose_56 = x_201.transpose(1, 2)
        x_201 = None
        x_202 = transpose_56.reshape(1, 257, 1024)
        transpose_56 = None
        x_203 = torch.nn.functional.layer_norm(
            x_202,
            (1024,),
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_202 = l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_203 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_205 = torch.nn.functional.dropout(x_204, 0.0, False, False)
        x_204 = None
        x_206 = x_199 + x_205
        x_199 = x_205 = None
        x_207 = torch.nn.functional.layer_norm(
            x_206,
            (1024,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        ) = None
        x_gate_13 = torch._C._nn.linear(
            x_207,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_208 = torch._C._nn.linear(
            x_207,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_207 = l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_13 = torch.nn.functional.silu(x_gate_13, inplace=False)
        x_gate_13 = None
        x_209 = silu_13 * x_208
        silu_13 = x_208 = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch.nn.functional.layer_norm(
            x_210,
            (2730,),
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_210 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_212 = torch._C._nn.linear(
            x_211,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_211 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_213 = torch.nn.functional.dropout(x_212, 0.0, False, False)
        x_212 = None
        x_214 = x_206 + x_213
        x_206 = x_213 = None
        x_215 = torch.nn.functional.layer_norm(
            x_214,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        ) = None
        linear_98 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_14_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_84 = linear_98.reshape(1, 257, 16, -1)
        linear_98 = None
        q_28 = reshape_84.transpose(1, 2)
        reshape_84 = None
        linear_99 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_14_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_85 = linear_99.reshape(1, 257, 16, -1)
        linear_99 = None
        k_28 = reshape_85.transpose(1, 2)
        reshape_85 = None
        linear_100 = torch._C._nn.linear(
            x_215,
            l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_215 = l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_14_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_86 = linear_100.reshape(1, 257, 16, -1)
        linear_100 = None
        v_14 = reshape_86.transpose(1, 2)
        reshape_86 = None
        getitem_172 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_173 = q_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_28 = None
        tensor_split_28 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_28 = tensor_split_28[0]
        cos_emb_28 = tensor_split_28[1]
        tensor_split_28 = None
        mul_70 = getitem_173 * cos_emb_28
        cos_emb_28 = None
        getitem_176 = getitem_173[(Ellipsis, slice(1, None, 2))]
        neg_28 = -getitem_176
        getitem_176 = None
        getitem_177 = getitem_173[(Ellipsis, slice(None, None, 2))]
        getitem_173 = None
        stack_28 = torch.stack([neg_28, getitem_177], -1)
        neg_28 = getitem_177 = None
        reshape_87 = stack_28.reshape((1, 16, 256, 64))
        stack_28 = None
        mul_71 = reshape_87 * sin_emb_28
        reshape_87 = sin_emb_28 = None
        add_57 = mul_70 + mul_71
        mul_70 = mul_71 = None
        cat_29 = torch.cat([getitem_172, add_57], dim=2)
        getitem_172 = add_57 = None
        q_29 = cat_29.type_as(v_14)
        cat_29 = None
        getitem_178 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_179 = k_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_28 = None
        tensor_split_29 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_29 = tensor_split_29[0]
        cos_emb_29 = tensor_split_29[1]
        tensor_split_29 = None
        mul_72 = getitem_179 * cos_emb_29
        cos_emb_29 = None
        getitem_182 = getitem_179[(Ellipsis, slice(1, None, 2))]
        neg_29 = -getitem_182
        getitem_182 = None
        getitem_183 = getitem_179[(Ellipsis, slice(None, None, 2))]
        getitem_179 = None
        stack_29 = torch.stack([neg_29, getitem_183], -1)
        neg_29 = getitem_183 = None
        reshape_88 = stack_29.reshape((1, 16, 256, 64))
        stack_29 = None
        mul_73 = reshape_88 * sin_emb_29
        reshape_88 = sin_emb_29 = None
        add_58 = mul_72 + mul_73
        mul_72 = mul_73 = None
        cat_30 = torch.cat([getitem_178, add_58], dim=2)
        getitem_178 = add_58 = None
        k_29 = cat_30.type_as(v_14)
        cat_30 = None
        x_216 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_14, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_14 = None
        transpose_60 = x_216.transpose(1, 2)
        x_216 = None
        x_217 = transpose_60.reshape(1, 257, 1024)
        transpose_60 = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (1024,),
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_217 = l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_218 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_220 = torch.nn.functional.dropout(x_219, 0.0, False, False)
        x_219 = None
        x_221 = x_214 + x_220
        x_214 = x_220 = None
        x_222 = torch.nn.functional.layer_norm(
            x_221,
            (1024,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        ) = None
        x_gate_14 = torch._C._nn.linear(
            x_222,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_223 = torch._C._nn.linear(
            x_222,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_222 = l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_14 = torch.nn.functional.silu(x_gate_14, inplace=False)
        x_gate_14 = None
        x_224 = silu_14 * x_223
        silu_14 = x_223 = None
        x_225 = torch.nn.functional.dropout(x_224, 0.0, False, False)
        x_224 = None
        x_226 = torch.nn.functional.layer_norm(
            x_225,
            (2730,),
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_225 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_227 = torch._C._nn.linear(
            x_226,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_226 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_228 = torch.nn.functional.dropout(x_227, 0.0, False, False)
        x_227 = None
        x_229 = x_221 + x_228
        x_221 = x_228 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        ) = None
        linear_105 = torch._C._nn.linear(
            x_230,
            l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_15_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_90 = linear_105.reshape(1, 257, 16, -1)
        linear_105 = None
        q_30 = reshape_90.transpose(1, 2)
        reshape_90 = None
        linear_106 = torch._C._nn.linear(
            x_230,
            l_self_modules_blocks_modules_15_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_91 = linear_106.reshape(1, 257, 16, -1)
        linear_106 = None
        k_30 = reshape_91.transpose(1, 2)
        reshape_91 = None
        linear_107 = torch._C._nn.linear(
            x_230,
            l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_230 = l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_15_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_92 = linear_107.reshape(1, 257, 16, -1)
        linear_107 = None
        v_15 = reshape_92.transpose(1, 2)
        reshape_92 = None
        getitem_184 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_185 = q_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_30 = None
        tensor_split_30 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_30 = tensor_split_30[0]
        cos_emb_30 = tensor_split_30[1]
        tensor_split_30 = None
        mul_75 = getitem_185 * cos_emb_30
        cos_emb_30 = None
        getitem_188 = getitem_185[(Ellipsis, slice(1, None, 2))]
        neg_30 = -getitem_188
        getitem_188 = None
        getitem_189 = getitem_185[(Ellipsis, slice(None, None, 2))]
        getitem_185 = None
        stack_30 = torch.stack([neg_30, getitem_189], -1)
        neg_30 = getitem_189 = None
        reshape_93 = stack_30.reshape((1, 16, 256, 64))
        stack_30 = None
        mul_76 = reshape_93 * sin_emb_30
        reshape_93 = sin_emb_30 = None
        add_61 = mul_75 + mul_76
        mul_75 = mul_76 = None
        cat_31 = torch.cat([getitem_184, add_61], dim=2)
        getitem_184 = add_61 = None
        q_31 = cat_31.type_as(v_15)
        cat_31 = None
        getitem_190 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_191 = k_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_30 = None
        tensor_split_31 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_31 = tensor_split_31[0]
        cos_emb_31 = tensor_split_31[1]
        tensor_split_31 = None
        mul_77 = getitem_191 * cos_emb_31
        cos_emb_31 = None
        getitem_194 = getitem_191[(Ellipsis, slice(1, None, 2))]
        neg_31 = -getitem_194
        getitem_194 = None
        getitem_195 = getitem_191[(Ellipsis, slice(None, None, 2))]
        getitem_191 = None
        stack_31 = torch.stack([neg_31, getitem_195], -1)
        neg_31 = getitem_195 = None
        reshape_94 = stack_31.reshape((1, 16, 256, 64))
        stack_31 = None
        mul_78 = reshape_94 * sin_emb_31
        reshape_94 = sin_emb_31 = None
        add_62 = mul_77 + mul_78
        mul_77 = mul_78 = None
        cat_32 = torch.cat([getitem_190, add_62], dim=2)
        getitem_190 = add_62 = None
        k_31 = cat_32.type_as(v_15)
        cat_32 = None
        x_231 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_15, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_15 = None
        transpose_64 = x_231.transpose(1, 2)
        x_231 = None
        x_232 = transpose_64.reshape(1, 257, 1024)
        transpose_64 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (1024,),
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_232 = l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_234 = torch._C._nn.linear(
            x_233,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_233 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_235 = torch.nn.functional.dropout(x_234, 0.0, False, False)
        x_234 = None
        x_236 = x_229 + x_235
        x_229 = x_235 = None
        x_237 = torch.nn.functional.layer_norm(
            x_236,
            (1024,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        ) = None
        x_gate_15 = torch._C._nn.linear(
            x_237,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_238 = torch._C._nn.linear(
            x_237,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_237 = l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_15 = torch.nn.functional.silu(x_gate_15, inplace=False)
        x_gate_15 = None
        x_239 = silu_15 * x_238
        silu_15 = x_238 = None
        x_240 = torch.nn.functional.dropout(x_239, 0.0, False, False)
        x_239 = None
        x_241 = torch.nn.functional.layer_norm(
            x_240,
            (2730,),
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_240 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_242 = torch._C._nn.linear(
            x_241,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_241 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_243 = torch.nn.functional.dropout(x_242, 0.0, False, False)
        x_242 = None
        x_244 = x_236 + x_243
        x_236 = x_243 = None
        x_245 = torch.nn.functional.layer_norm(
            x_244,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        ) = None
        linear_112 = torch._C._nn.linear(
            x_245,
            l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_16_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_96 = linear_112.reshape(1, 257, 16, -1)
        linear_112 = None
        q_32 = reshape_96.transpose(1, 2)
        reshape_96 = None
        linear_113 = torch._C._nn.linear(
            x_245,
            l_self_modules_blocks_modules_16_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_97 = linear_113.reshape(1, 257, 16, -1)
        linear_113 = None
        k_32 = reshape_97.transpose(1, 2)
        reshape_97 = None
        linear_114 = torch._C._nn.linear(
            x_245,
            l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_245 = l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_16_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_98 = linear_114.reshape(1, 257, 16, -1)
        linear_114 = None
        v_16 = reshape_98.transpose(1, 2)
        reshape_98 = None
        getitem_196 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_197 = q_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_32 = None
        tensor_split_32 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_32 = tensor_split_32[0]
        cos_emb_32 = tensor_split_32[1]
        tensor_split_32 = None
        mul_80 = getitem_197 * cos_emb_32
        cos_emb_32 = None
        getitem_200 = getitem_197[(Ellipsis, slice(1, None, 2))]
        neg_32 = -getitem_200
        getitem_200 = None
        getitem_201 = getitem_197[(Ellipsis, slice(None, None, 2))]
        getitem_197 = None
        stack_32 = torch.stack([neg_32, getitem_201], -1)
        neg_32 = getitem_201 = None
        reshape_99 = stack_32.reshape((1, 16, 256, 64))
        stack_32 = None
        mul_81 = reshape_99 * sin_emb_32
        reshape_99 = sin_emb_32 = None
        add_65 = mul_80 + mul_81
        mul_80 = mul_81 = None
        cat_33 = torch.cat([getitem_196, add_65], dim=2)
        getitem_196 = add_65 = None
        q_33 = cat_33.type_as(v_16)
        cat_33 = None
        getitem_202 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_203 = k_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_32 = None
        tensor_split_33 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_33 = tensor_split_33[0]
        cos_emb_33 = tensor_split_33[1]
        tensor_split_33 = None
        mul_82 = getitem_203 * cos_emb_33
        cos_emb_33 = None
        getitem_206 = getitem_203[(Ellipsis, slice(1, None, 2))]
        neg_33 = -getitem_206
        getitem_206 = None
        getitem_207 = getitem_203[(Ellipsis, slice(None, None, 2))]
        getitem_203 = None
        stack_33 = torch.stack([neg_33, getitem_207], -1)
        neg_33 = getitem_207 = None
        reshape_100 = stack_33.reshape((1, 16, 256, 64))
        stack_33 = None
        mul_83 = reshape_100 * sin_emb_33
        reshape_100 = sin_emb_33 = None
        add_66 = mul_82 + mul_83
        mul_82 = mul_83 = None
        cat_34 = torch.cat([getitem_202, add_66], dim=2)
        getitem_202 = add_66 = None
        k_33 = cat_34.type_as(v_16)
        cat_34 = None
        x_246 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_16, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_16 = None
        transpose_68 = x_246.transpose(1, 2)
        x_246 = None
        x_247 = transpose_68.reshape(1, 257, 1024)
        transpose_68 = None
        x_248 = torch.nn.functional.layer_norm(
            x_247,
            (1024,),
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_247 = l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_248 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_250 = torch.nn.functional.dropout(x_249, 0.0, False, False)
        x_249 = None
        x_251 = x_244 + x_250
        x_244 = x_250 = None
        x_252 = torch.nn.functional.layer_norm(
            x_251,
            (1024,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        ) = None
        x_gate_16 = torch._C._nn.linear(
            x_252,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_252 = l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_16 = torch.nn.functional.silu(x_gate_16, inplace=False)
        x_gate_16 = None
        x_254 = silu_16 * x_253
        silu_16 = x_253 = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (2730,),
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_255 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_257 = torch._C._nn.linear(
            x_256,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_256 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_258 = torch.nn.functional.dropout(x_257, 0.0, False, False)
        x_257 = None
        x_259 = x_251 + x_258
        x_251 = x_258 = None
        x_260 = torch.nn.functional.layer_norm(
            x_259,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        ) = None
        linear_119 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_17_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_102 = linear_119.reshape(1, 257, 16, -1)
        linear_119 = None
        q_34 = reshape_102.transpose(1, 2)
        reshape_102 = None
        linear_120 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_17_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_103 = linear_120.reshape(1, 257, 16, -1)
        linear_120 = None
        k_34 = reshape_103.transpose(1, 2)
        reshape_103 = None
        linear_121 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_260 = l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_17_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_104 = linear_121.reshape(1, 257, 16, -1)
        linear_121 = None
        v_17 = reshape_104.transpose(1, 2)
        reshape_104 = None
        getitem_208 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_209 = q_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_34 = None
        tensor_split_34 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_34 = tensor_split_34[0]
        cos_emb_34 = tensor_split_34[1]
        tensor_split_34 = None
        mul_85 = getitem_209 * cos_emb_34
        cos_emb_34 = None
        getitem_212 = getitem_209[(Ellipsis, slice(1, None, 2))]
        neg_34 = -getitem_212
        getitem_212 = None
        getitem_213 = getitem_209[(Ellipsis, slice(None, None, 2))]
        getitem_209 = None
        stack_34 = torch.stack([neg_34, getitem_213], -1)
        neg_34 = getitem_213 = None
        reshape_105 = stack_34.reshape((1, 16, 256, 64))
        stack_34 = None
        mul_86 = reshape_105 * sin_emb_34
        reshape_105 = sin_emb_34 = None
        add_69 = mul_85 + mul_86
        mul_85 = mul_86 = None
        cat_35 = torch.cat([getitem_208, add_69], dim=2)
        getitem_208 = add_69 = None
        q_35 = cat_35.type_as(v_17)
        cat_35 = None
        getitem_214 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_215 = k_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_34 = None
        tensor_split_35 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_35 = tensor_split_35[0]
        cos_emb_35 = tensor_split_35[1]
        tensor_split_35 = None
        mul_87 = getitem_215 * cos_emb_35
        cos_emb_35 = None
        getitem_218 = getitem_215[(Ellipsis, slice(1, None, 2))]
        neg_35 = -getitem_218
        getitem_218 = None
        getitem_219 = getitem_215[(Ellipsis, slice(None, None, 2))]
        getitem_215 = None
        stack_35 = torch.stack([neg_35, getitem_219], -1)
        neg_35 = getitem_219 = None
        reshape_106 = stack_35.reshape((1, 16, 256, 64))
        stack_35 = None
        mul_88 = reshape_106 * sin_emb_35
        reshape_106 = sin_emb_35 = None
        add_70 = mul_87 + mul_88
        mul_87 = mul_88 = None
        cat_36 = torch.cat([getitem_214, add_70], dim=2)
        getitem_214 = add_70 = None
        k_35 = cat_36.type_as(v_17)
        cat_36 = None
        x_261 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_17, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_17 = None
        transpose_72 = x_261.transpose(1, 2)
        x_261 = None
        x_262 = transpose_72.reshape(1, 257, 1024)
        transpose_72 = None
        x_263 = torch.nn.functional.layer_norm(
            x_262,
            (1024,),
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_262 = l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_264 = torch._C._nn.linear(
            x_263,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_263 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_265 = torch.nn.functional.dropout(x_264, 0.0, False, False)
        x_264 = None
        x_266 = x_259 + x_265
        x_259 = x_265 = None
        x_267 = torch.nn.functional.layer_norm(
            x_266,
            (1024,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        ) = None
        x_gate_17 = torch._C._nn.linear(
            x_267,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_267 = l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_17 = torch.nn.functional.silu(x_gate_17, inplace=False)
        x_gate_17 = None
        x_269 = silu_17 * x_268
        silu_17 = x_268 = None
        x_270 = torch.nn.functional.dropout(x_269, 0.0, False, False)
        x_269 = None
        x_271 = torch.nn.functional.layer_norm(
            x_270,
            (2730,),
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_270 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_271 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = x_266 + x_273
        x_266 = x_273 = None
        x_275 = torch.nn.functional.layer_norm(
            x_274,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        ) = None
        linear_126 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_18_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_108 = linear_126.reshape(1, 257, 16, -1)
        linear_126 = None
        q_36 = reshape_108.transpose(1, 2)
        reshape_108 = None
        linear_127 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_18_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_109 = linear_127.reshape(1, 257, 16, -1)
        linear_127 = None
        k_36 = reshape_109.transpose(1, 2)
        reshape_109 = None
        linear_128 = torch._C._nn.linear(
            x_275,
            l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_275 = l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_18_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_110 = linear_128.reshape(1, 257, 16, -1)
        linear_128 = None
        v_18 = reshape_110.transpose(1, 2)
        reshape_110 = None
        getitem_220 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_221 = q_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_36 = None
        tensor_split_36 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_36 = tensor_split_36[0]
        cos_emb_36 = tensor_split_36[1]
        tensor_split_36 = None
        mul_90 = getitem_221 * cos_emb_36
        cos_emb_36 = None
        getitem_224 = getitem_221[(Ellipsis, slice(1, None, 2))]
        neg_36 = -getitem_224
        getitem_224 = None
        getitem_225 = getitem_221[(Ellipsis, slice(None, None, 2))]
        getitem_221 = None
        stack_36 = torch.stack([neg_36, getitem_225], -1)
        neg_36 = getitem_225 = None
        reshape_111 = stack_36.reshape((1, 16, 256, 64))
        stack_36 = None
        mul_91 = reshape_111 * sin_emb_36
        reshape_111 = sin_emb_36 = None
        add_73 = mul_90 + mul_91
        mul_90 = mul_91 = None
        cat_37 = torch.cat([getitem_220, add_73], dim=2)
        getitem_220 = add_73 = None
        q_37 = cat_37.type_as(v_18)
        cat_37 = None
        getitem_226 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_227 = k_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_36 = None
        tensor_split_37 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_37 = tensor_split_37[0]
        cos_emb_37 = tensor_split_37[1]
        tensor_split_37 = None
        mul_92 = getitem_227 * cos_emb_37
        cos_emb_37 = None
        getitem_230 = getitem_227[(Ellipsis, slice(1, None, 2))]
        neg_37 = -getitem_230
        getitem_230 = None
        getitem_231 = getitem_227[(Ellipsis, slice(None, None, 2))]
        getitem_227 = None
        stack_37 = torch.stack([neg_37, getitem_231], -1)
        neg_37 = getitem_231 = None
        reshape_112 = stack_37.reshape((1, 16, 256, 64))
        stack_37 = None
        mul_93 = reshape_112 * sin_emb_37
        reshape_112 = sin_emb_37 = None
        add_74 = mul_92 + mul_93
        mul_92 = mul_93 = None
        cat_38 = torch.cat([getitem_226, add_74], dim=2)
        getitem_226 = add_74 = None
        k_37 = cat_38.type_as(v_18)
        cat_38 = None
        x_276 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_18, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_18 = None
        transpose_76 = x_276.transpose(1, 2)
        x_276 = None
        x_277 = transpose_76.reshape(1, 257, 1024)
        transpose_76 = None
        x_278 = torch.nn.functional.layer_norm(
            x_277,
            (1024,),
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_277 = l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_279 = torch._C._nn.linear(
            x_278,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_278 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_280 = torch.nn.functional.dropout(x_279, 0.0, False, False)
        x_279 = None
        x_281 = x_274 + x_280
        x_274 = x_280 = None
        x_282 = torch.nn.functional.layer_norm(
            x_281,
            (1024,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        ) = None
        x_gate_18 = torch._C._nn.linear(
            x_282,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_283 = torch._C._nn.linear(
            x_282,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_282 = l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_18 = torch.nn.functional.silu(x_gate_18, inplace=False)
        x_gate_18 = None
        x_284 = silu_18 * x_283
        silu_18 = x_283 = None
        x_285 = torch.nn.functional.dropout(x_284, 0.0, False, False)
        x_284 = None
        x_286 = torch.nn.functional.layer_norm(
            x_285,
            (2730,),
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_285 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = x_281 + x_288
        x_281 = x_288 = None
        x_290 = torch.nn.functional.layer_norm(
            x_289,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        ) = None
        linear_133 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_19_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_114 = linear_133.reshape(1, 257, 16, -1)
        linear_133 = None
        q_38 = reshape_114.transpose(1, 2)
        reshape_114 = None
        linear_134 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_19_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_115 = linear_134.reshape(1, 257, 16, -1)
        linear_134 = None
        k_38 = reshape_115.transpose(1, 2)
        reshape_115 = None
        linear_135 = torch._C._nn.linear(
            x_290,
            l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_290 = l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_19_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_116 = linear_135.reshape(1, 257, 16, -1)
        linear_135 = None
        v_19 = reshape_116.transpose(1, 2)
        reshape_116 = None
        getitem_232 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_233 = q_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_38 = None
        tensor_split_38 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_38 = tensor_split_38[0]
        cos_emb_38 = tensor_split_38[1]
        tensor_split_38 = None
        mul_95 = getitem_233 * cos_emb_38
        cos_emb_38 = None
        getitem_236 = getitem_233[(Ellipsis, slice(1, None, 2))]
        neg_38 = -getitem_236
        getitem_236 = None
        getitem_237 = getitem_233[(Ellipsis, slice(None, None, 2))]
        getitem_233 = None
        stack_38 = torch.stack([neg_38, getitem_237], -1)
        neg_38 = getitem_237 = None
        reshape_117 = stack_38.reshape((1, 16, 256, 64))
        stack_38 = None
        mul_96 = reshape_117 * sin_emb_38
        reshape_117 = sin_emb_38 = None
        add_77 = mul_95 + mul_96
        mul_95 = mul_96 = None
        cat_39 = torch.cat([getitem_232, add_77], dim=2)
        getitem_232 = add_77 = None
        q_39 = cat_39.type_as(v_19)
        cat_39 = None
        getitem_238 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_239 = k_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_38 = None
        tensor_split_39 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_39 = tensor_split_39[0]
        cos_emb_39 = tensor_split_39[1]
        tensor_split_39 = None
        mul_97 = getitem_239 * cos_emb_39
        cos_emb_39 = None
        getitem_242 = getitem_239[(Ellipsis, slice(1, None, 2))]
        neg_39 = -getitem_242
        getitem_242 = None
        getitem_243 = getitem_239[(Ellipsis, slice(None, None, 2))]
        getitem_239 = None
        stack_39 = torch.stack([neg_39, getitem_243], -1)
        neg_39 = getitem_243 = None
        reshape_118 = stack_39.reshape((1, 16, 256, 64))
        stack_39 = None
        mul_98 = reshape_118 * sin_emb_39
        reshape_118 = sin_emb_39 = None
        add_78 = mul_97 + mul_98
        mul_97 = mul_98 = None
        cat_40 = torch.cat([getitem_238, add_78], dim=2)
        getitem_238 = add_78 = None
        k_39 = cat_40.type_as(v_19)
        cat_40 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_19, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_19 = None
        transpose_80 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_80.reshape(1, 257, 1024)
        transpose_80 = None
        x_293 = torch.nn.functional.layer_norm(
            x_292,
            (1024,),
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_292 = l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_293 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_295 = torch.nn.functional.dropout(x_294, 0.0, False, False)
        x_294 = None
        x_296 = x_289 + x_295
        x_289 = x_295 = None
        x_297 = torch.nn.functional.layer_norm(
            x_296,
            (1024,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        ) = None
        x_gate_19 = torch._C._nn.linear(
            x_297,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_297 = l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_19 = torch.nn.functional.silu(x_gate_19, inplace=False)
        x_gate_19 = None
        x_299 = silu_19 * x_298
        silu_19 = x_298 = None
        x_300 = torch.nn.functional.dropout(x_299, 0.0, False, False)
        x_299 = None
        x_301 = torch.nn.functional.layer_norm(
            x_300,
            (2730,),
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_300 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_302 = torch._C._nn.linear(
            x_301,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_301 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_303 = torch.nn.functional.dropout(x_302, 0.0, False, False)
        x_302 = None
        x_304 = x_296 + x_303
        x_296 = x_303 = None
        x_305 = torch.nn.functional.layer_norm(
            x_304,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        ) = None
        linear_140 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_20_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_120 = linear_140.reshape(1, 257, 16, -1)
        linear_140 = None
        q_40 = reshape_120.transpose(1, 2)
        reshape_120 = None
        linear_141 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_20_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_121 = linear_141.reshape(1, 257, 16, -1)
        linear_141 = None
        k_40 = reshape_121.transpose(1, 2)
        reshape_121 = None
        linear_142 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_305 = l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_20_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_122 = linear_142.reshape(1, 257, 16, -1)
        linear_142 = None
        v_20 = reshape_122.transpose(1, 2)
        reshape_122 = None
        getitem_244 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_245 = q_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_40 = None
        tensor_split_40 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_40 = tensor_split_40[0]
        cos_emb_40 = tensor_split_40[1]
        tensor_split_40 = None
        mul_100 = getitem_245 * cos_emb_40
        cos_emb_40 = None
        getitem_248 = getitem_245[(Ellipsis, slice(1, None, 2))]
        neg_40 = -getitem_248
        getitem_248 = None
        getitem_249 = getitem_245[(Ellipsis, slice(None, None, 2))]
        getitem_245 = None
        stack_40 = torch.stack([neg_40, getitem_249], -1)
        neg_40 = getitem_249 = None
        reshape_123 = stack_40.reshape((1, 16, 256, 64))
        stack_40 = None
        mul_101 = reshape_123 * sin_emb_40
        reshape_123 = sin_emb_40 = None
        add_81 = mul_100 + mul_101
        mul_100 = mul_101 = None
        cat_41 = torch.cat([getitem_244, add_81], dim=2)
        getitem_244 = add_81 = None
        q_41 = cat_41.type_as(v_20)
        cat_41 = None
        getitem_250 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_251 = k_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_40 = None
        tensor_split_41 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_41 = tensor_split_41[0]
        cos_emb_41 = tensor_split_41[1]
        tensor_split_41 = None
        mul_102 = getitem_251 * cos_emb_41
        cos_emb_41 = None
        getitem_254 = getitem_251[(Ellipsis, slice(1, None, 2))]
        neg_41 = -getitem_254
        getitem_254 = None
        getitem_255 = getitem_251[(Ellipsis, slice(None, None, 2))]
        getitem_251 = None
        stack_41 = torch.stack([neg_41, getitem_255], -1)
        neg_41 = getitem_255 = None
        reshape_124 = stack_41.reshape((1, 16, 256, 64))
        stack_41 = None
        mul_103 = reshape_124 * sin_emb_41
        reshape_124 = sin_emb_41 = None
        add_82 = mul_102 + mul_103
        mul_102 = mul_103 = None
        cat_42 = torch.cat([getitem_250, add_82], dim=2)
        getitem_250 = add_82 = None
        k_41 = cat_42.type_as(v_20)
        cat_42 = None
        x_306 = torch._C._nn.scaled_dot_product_attention(
            q_41, k_41, v_20, attn_mask=None, dropout_p=0.0
        )
        q_41 = k_41 = v_20 = None
        transpose_84 = x_306.transpose(1, 2)
        x_306 = None
        x_307 = transpose_84.reshape(1, 257, 1024)
        transpose_84 = None
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (1024,),
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_307 = l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_308 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_310 = torch.nn.functional.dropout(x_309, 0.0, False, False)
        x_309 = None
        x_311 = x_304 + x_310
        x_304 = x_310 = None
        x_312 = torch.nn.functional.layer_norm(
            x_311,
            (1024,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        ) = None
        x_gate_20 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_312 = l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_20 = torch.nn.functional.silu(x_gate_20, inplace=False)
        x_gate_20 = None
        x_314 = silu_20 * x_313
        silu_20 = x_313 = None
        x_315 = torch.nn.functional.dropout(x_314, 0.0, False, False)
        x_314 = None
        x_316 = torch.nn.functional.layer_norm(
            x_315,
            (2730,),
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_315 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_317 = torch._C._nn.linear(
            x_316,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_316 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_318 = torch.nn.functional.dropout(x_317, 0.0, False, False)
        x_317 = None
        x_319 = x_311 + x_318
        x_311 = x_318 = None
        x_320 = torch.nn.functional.layer_norm(
            x_319,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        ) = None
        linear_147 = torch._C._nn.linear(
            x_320,
            l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_21_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_126 = linear_147.reshape(1, 257, 16, -1)
        linear_147 = None
        q_42 = reshape_126.transpose(1, 2)
        reshape_126 = None
        linear_148 = torch._C._nn.linear(
            x_320,
            l_self_modules_blocks_modules_21_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_127 = linear_148.reshape(1, 257, 16, -1)
        linear_148 = None
        k_42 = reshape_127.transpose(1, 2)
        reshape_127 = None
        linear_149 = torch._C._nn.linear(
            x_320,
            l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_320 = l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_21_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_128 = linear_149.reshape(1, 257, 16, -1)
        linear_149 = None
        v_21 = reshape_128.transpose(1, 2)
        reshape_128 = None
        getitem_256 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_257 = q_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_42 = None
        tensor_split_42 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_42 = tensor_split_42[0]
        cos_emb_42 = tensor_split_42[1]
        tensor_split_42 = None
        mul_105 = getitem_257 * cos_emb_42
        cos_emb_42 = None
        getitem_260 = getitem_257[(Ellipsis, slice(1, None, 2))]
        neg_42 = -getitem_260
        getitem_260 = None
        getitem_261 = getitem_257[(Ellipsis, slice(None, None, 2))]
        getitem_257 = None
        stack_42 = torch.stack([neg_42, getitem_261], -1)
        neg_42 = getitem_261 = None
        reshape_129 = stack_42.reshape((1, 16, 256, 64))
        stack_42 = None
        mul_106 = reshape_129 * sin_emb_42
        reshape_129 = sin_emb_42 = None
        add_85 = mul_105 + mul_106
        mul_105 = mul_106 = None
        cat_43 = torch.cat([getitem_256, add_85], dim=2)
        getitem_256 = add_85 = None
        q_43 = cat_43.type_as(v_21)
        cat_43 = None
        getitem_262 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_263 = k_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_42 = None
        tensor_split_43 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_43 = tensor_split_43[0]
        cos_emb_43 = tensor_split_43[1]
        tensor_split_43 = None
        mul_107 = getitem_263 * cos_emb_43
        cos_emb_43 = None
        getitem_266 = getitem_263[(Ellipsis, slice(1, None, 2))]
        neg_43 = -getitem_266
        getitem_266 = None
        getitem_267 = getitem_263[(Ellipsis, slice(None, None, 2))]
        getitem_263 = None
        stack_43 = torch.stack([neg_43, getitem_267], -1)
        neg_43 = getitem_267 = None
        reshape_130 = stack_43.reshape((1, 16, 256, 64))
        stack_43 = None
        mul_108 = reshape_130 * sin_emb_43
        reshape_130 = sin_emb_43 = None
        add_86 = mul_107 + mul_108
        mul_107 = mul_108 = None
        cat_44 = torch.cat([getitem_262, add_86], dim=2)
        getitem_262 = add_86 = None
        k_43 = cat_44.type_as(v_21)
        cat_44 = None
        x_321 = torch._C._nn.scaled_dot_product_attention(
            q_43, k_43, v_21, attn_mask=None, dropout_p=0.0
        )
        q_43 = k_43 = v_21 = None
        transpose_88 = x_321.transpose(1, 2)
        x_321 = None
        x_322 = transpose_88.reshape(1, 257, 1024)
        transpose_88 = None
        x_323 = torch.nn.functional.layer_norm(
            x_322,
            (1024,),
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_322 = l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_323 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = x_319 + x_325
        x_319 = x_325 = None
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (1024,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        ) = None
        x_gate_21 = torch._C._nn.linear(
            x_327,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_328 = torch._C._nn.linear(
            x_327,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_327 = l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_21 = torch.nn.functional.silu(x_gate_21, inplace=False)
        x_gate_21 = None
        x_329 = silu_21 * x_328
        silu_21 = x_328 = None
        x_330 = torch.nn.functional.dropout(x_329, 0.0, False, False)
        x_329 = None
        x_331 = torch.nn.functional.layer_norm(
            x_330,
            (2730,),
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_330 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_331 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = x_326 + x_333
        x_326 = x_333 = None
        x_335 = torch.nn.functional.layer_norm(
            x_334,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        ) = None
        linear_154 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_22_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_132 = linear_154.reshape(1, 257, 16, -1)
        linear_154 = None
        q_44 = reshape_132.transpose(1, 2)
        reshape_132 = None
        linear_155 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_22_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_133 = linear_155.reshape(1, 257, 16, -1)
        linear_155 = None
        k_44 = reshape_133.transpose(1, 2)
        reshape_133 = None
        linear_156 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_335 = l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_22_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_134 = linear_156.reshape(1, 257, 16, -1)
        linear_156 = None
        v_22 = reshape_134.transpose(1, 2)
        reshape_134 = None
        getitem_268 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_269 = q_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_44 = None
        tensor_split_44 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_44 = tensor_split_44[0]
        cos_emb_44 = tensor_split_44[1]
        tensor_split_44 = None
        mul_110 = getitem_269 * cos_emb_44
        cos_emb_44 = None
        getitem_272 = getitem_269[(Ellipsis, slice(1, None, 2))]
        neg_44 = -getitem_272
        getitem_272 = None
        getitem_273 = getitem_269[(Ellipsis, slice(None, None, 2))]
        getitem_269 = None
        stack_44 = torch.stack([neg_44, getitem_273], -1)
        neg_44 = getitem_273 = None
        reshape_135 = stack_44.reshape((1, 16, 256, 64))
        stack_44 = None
        mul_111 = reshape_135 * sin_emb_44
        reshape_135 = sin_emb_44 = None
        add_89 = mul_110 + mul_111
        mul_110 = mul_111 = None
        cat_45 = torch.cat([getitem_268, add_89], dim=2)
        getitem_268 = add_89 = None
        q_45 = cat_45.type_as(v_22)
        cat_45 = None
        getitem_274 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_275 = k_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_44 = None
        tensor_split_45 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_45 = tensor_split_45[0]
        cos_emb_45 = tensor_split_45[1]
        tensor_split_45 = None
        mul_112 = getitem_275 * cos_emb_45
        cos_emb_45 = None
        getitem_278 = getitem_275[(Ellipsis, slice(1, None, 2))]
        neg_45 = -getitem_278
        getitem_278 = None
        getitem_279 = getitem_275[(Ellipsis, slice(None, None, 2))]
        getitem_275 = None
        stack_45 = torch.stack([neg_45, getitem_279], -1)
        neg_45 = getitem_279 = None
        reshape_136 = stack_45.reshape((1, 16, 256, 64))
        stack_45 = None
        mul_113 = reshape_136 * sin_emb_45
        reshape_136 = sin_emb_45 = None
        add_90 = mul_112 + mul_113
        mul_112 = mul_113 = None
        cat_46 = torch.cat([getitem_274, add_90], dim=2)
        getitem_274 = add_90 = None
        k_45 = cat_46.type_as(v_22)
        cat_46 = None
        x_336 = torch._C._nn.scaled_dot_product_attention(
            q_45, k_45, v_22, attn_mask=None, dropout_p=0.0
        )
        q_45 = k_45 = v_22 = None
        transpose_92 = x_336.transpose(1, 2)
        x_336 = None
        x_337 = transpose_92.reshape(1, 257, 1024)
        transpose_92 = None
        x_338 = torch.nn.functional.layer_norm(
            x_337,
            (1024,),
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_337 = l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_338 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.dropout(x_339, 0.0, False, False)
        x_339 = None
        x_341 = x_334 + x_340
        x_334 = x_340 = None
        x_342 = torch.nn.functional.layer_norm(
            x_341,
            (1024,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        ) = None
        x_gate_22 = torch._C._nn.linear(
            x_342,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_343 = torch._C._nn.linear(
            x_342,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_342 = l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_22 = torch.nn.functional.silu(x_gate_22, inplace=False)
        x_gate_22 = None
        x_344 = silu_22 * x_343
        silu_22 = x_343 = None
        x_345 = torch.nn.functional.dropout(x_344, 0.0, False, False)
        x_344 = None
        x_346 = torch.nn.functional.layer_norm(
            x_345,
            (2730,),
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_345 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_347 = torch._C._nn.linear(
            x_346,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_346 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_348 = torch.nn.functional.dropout(x_347, 0.0, False, False)
        x_347 = None
        x_349 = x_341 + x_348
        x_341 = x_348 = None
        x_350 = torch.nn.functional.layer_norm(
            x_349,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        ) = None
        linear_161 = torch._C._nn.linear(
            x_350,
            l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_weight_ = l_self_modules_blocks_modules_23_modules_attn_modules_q_proj_parameters_bias_ = (None)
        reshape_138 = linear_161.reshape(1, 257, 16, -1)
        linear_161 = None
        q_46 = reshape_138.transpose(1, 2)
        reshape_138 = None
        linear_162 = torch._C._nn.linear(
            x_350,
            l_self_modules_blocks_modules_23_modules_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        reshape_139 = linear_162.reshape(1, 257, 16, -1)
        linear_162 = None
        k_46 = reshape_139.transpose(1, 2)
        reshape_139 = None
        linear_163 = torch._C._nn.linear(
            x_350,
            l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_bias_,
        )
        x_350 = l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_weight_ = l_self_modules_blocks_modules_23_modules_attn_modules_v_proj_parameters_bias_ = (None)
        reshape_140 = linear_163.reshape(1, 257, 16, -1)
        linear_163 = None
        v_23 = reshape_140.transpose(1, 2)
        reshape_140 = None
        getitem_280 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_281 = q_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        q_46 = None
        tensor_split_46 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        sin_emb_46 = tensor_split_46[0]
        cos_emb_46 = tensor_split_46[1]
        tensor_split_46 = None
        mul_115 = getitem_281 * cos_emb_46
        cos_emb_46 = None
        getitem_284 = getitem_281[(Ellipsis, slice(1, None, 2))]
        neg_46 = -getitem_284
        getitem_284 = None
        getitem_285 = getitem_281[(Ellipsis, slice(None, None, 2))]
        getitem_281 = None
        stack_46 = torch.stack([neg_46, getitem_285], -1)
        neg_46 = getitem_285 = None
        reshape_141 = stack_46.reshape((1, 16, 256, 64))
        stack_46 = None
        mul_116 = reshape_141 * sin_emb_46
        reshape_141 = sin_emb_46 = None
        add_93 = mul_115 + mul_116
        mul_115 = mul_116 = None
        cat_47 = torch.cat([getitem_280, add_93], dim=2)
        getitem_280 = add_93 = None
        q_47 = cat_47.type_as(v_23)
        cat_47 = None
        getitem_286 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 1, None),
                slice(None, None, None),
            )
        ]
        getitem_287 = k_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(1, None, None),
                slice(None, None, None),
            )
        ]
        k_46 = None
        tensor_split_47 = l_self_modules_rope_buffers_pos_embed_.tensor_split(2, -1)
        l_self_modules_rope_buffers_pos_embed_ = None
        sin_emb_47 = tensor_split_47[0]
        cos_emb_47 = tensor_split_47[1]
        tensor_split_47 = None
        mul_117 = getitem_287 * cos_emb_47
        cos_emb_47 = None
        getitem_290 = getitem_287[(Ellipsis, slice(1, None, 2))]
        neg_47 = -getitem_290
        getitem_290 = None
        getitem_291 = getitem_287[(Ellipsis, slice(None, None, 2))]
        getitem_287 = None
        stack_47 = torch.stack([neg_47, getitem_291], -1)
        neg_47 = getitem_291 = None
        reshape_142 = stack_47.reshape((1, 16, 256, 64))
        stack_47 = None
        mul_118 = reshape_142 * sin_emb_47
        reshape_142 = sin_emb_47 = None
        add_94 = mul_117 + mul_118
        mul_117 = mul_118 = None
        cat_48 = torch.cat([getitem_286, add_94], dim=2)
        getitem_286 = add_94 = None
        k_47 = cat_48.type_as(v_23)
        cat_48 = None
        x_351 = torch._C._nn.scaled_dot_product_attention(
            q_47, k_47, v_23, attn_mask=None, dropout_p=0.0
        )
        q_47 = k_47 = v_23 = None
        transpose_96 = x_351.transpose(1, 2)
        x_351 = None
        x_352 = transpose_96.reshape(1, 257, 1024)
        transpose_96 = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (1024,),
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_,
            1e-06,
        )
        x_352 = l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_norm_parameters_bias_
        ) = None
        x_354 = torch._C._nn.linear(
            x_353,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_353 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_355 = torch.nn.functional.dropout(x_354, 0.0, False, False)
        x_354 = None
        x_356 = x_349 + x_355
        x_349 = x_355 = None
        x_357 = torch.nn.functional.layer_norm(
            x_356,
            (1024,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        ) = None
        x_gate_23 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_bias_,
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_g_parameters_bias_
        ) = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_bias_,
        )
        x_357 = l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_x_parameters_bias_
        ) = None
        silu_23 = torch.nn.functional.silu(x_gate_23, inplace=False)
        x_gate_23 = None
        x_359 = silu_23 * x_358
        silu_23 = x_358 = None
        x_360 = torch.nn.functional.dropout(x_359, 0.0, False, False)
        x_359 = None
        x_361 = torch.nn.functional.layer_norm(
            x_360,
            (2730,),
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_,
            1e-06,
        )
        x_360 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_norm_parameters_bias_
        ) = None
        x_362 = torch._C._nn.linear(
            x_361,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_361 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_363 = torch.nn.functional.dropout(x_362, 0.0, False, False)
        x_362 = None
        x_364 = x_356 + x_363
        x_356 = x_363 = None
        x_365 = torch.nn.functional.layer_norm(
            x_364,
            (1024,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_364 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_366 = x_365[(slice(None, None, None), 0)]
        x_365 = None
        x_367 = torch.nn.functional.dropout(x_366, 0.0, False, False)
        x_366 = None
        x_368 = torch._C._nn.linear(
            x_367,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_367 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_368,)
