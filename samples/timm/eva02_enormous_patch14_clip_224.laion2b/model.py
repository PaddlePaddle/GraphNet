import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_patch_embed_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_patch_embed_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_parameters_pos_embed_: torch.nn.parameter.Parameter,
        L_self_parameters_cls_token_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_24_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_25_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_26_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_27_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_28_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_28_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_29_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_29_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_30_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_30_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_31_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_32_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_32_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_33_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_33_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_34_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_34_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_35_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_35_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_36_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_36_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_37_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_37_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_38_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_38_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_39_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_39_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_40_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_40_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_41_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_41_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_42_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_42_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_43_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_43_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_44_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_44_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_45_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_45_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_46_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_46_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_47_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_47_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_48_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_48_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_49_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_49_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_50_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_50_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_51_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_51_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_52_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_52_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_53_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_53_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_54_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_54_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_55_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_55_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_56_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_56_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_57_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_57_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_58_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_58_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_59_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_59_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_60_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_60_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_61_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_61_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_62_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_62_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_attn_parameters_q_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_attn_buffers_k_bias_: torch.Tensor,
        L_self_modules_blocks_modules_63_modules_attn_parameters_v_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_attn_modules_qkv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_63_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_parameters_cls_token_ = L_self_parameters_cls_token_
        l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_24_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_24_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_25_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_25_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_26_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_26_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_27_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_27_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_28_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_28_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_29_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_29_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_30_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_30_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_32_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_32_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_33_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_33_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_34_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_34_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_35_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_35_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_36_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_36_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_37_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_37_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_38_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_38_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_39_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_39_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_40_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_40_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_41_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_41_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_42_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_42_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_43_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_43_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_44_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_44_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_45_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_45_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_46_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_46_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_47_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_47_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_48_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_48_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_48_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_48_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_48_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_48_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_48_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_48_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_48_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_48_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_48_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_48_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_48_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_48_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_48_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_48_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_49_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_49_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_49_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_49_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_49_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_49_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_49_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_49_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_49_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_49_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_49_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_49_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_49_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_49_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_49_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_49_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_50_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_50_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_50_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_50_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_50_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_50_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_50_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_50_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_50_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_50_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_50_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_50_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_50_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_50_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_50_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_50_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_51_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_51_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_51_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_51_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_51_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_51_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_51_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_51_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_51_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_51_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_51_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_51_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_51_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_51_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_51_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_51_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_52_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_52_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_52_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_52_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_52_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_52_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_52_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_52_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_52_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_52_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_52_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_52_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_52_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_52_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_52_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_52_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_53_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_53_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_53_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_53_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_53_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_53_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_53_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_53_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_53_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_53_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_53_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_53_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_53_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_53_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_53_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_53_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_54_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_54_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_54_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_54_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_54_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_54_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_54_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_54_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_54_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_54_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_54_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_54_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_54_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_54_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_54_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_54_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_55_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_55_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_55_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_55_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_55_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_55_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_55_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_55_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_55_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_55_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_55_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_55_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_55_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_55_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_55_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_55_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_56_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_56_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_56_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_56_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_56_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_56_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_56_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_56_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_56_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_56_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_56_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_56_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_56_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_56_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_56_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_56_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_57_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_57_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_57_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_57_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_57_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_57_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_57_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_57_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_57_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_57_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_57_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_57_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_57_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_57_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_57_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_57_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_58_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_58_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_58_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_58_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_58_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_58_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_58_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_58_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_58_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_58_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_58_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_58_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_58_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_58_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_58_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_58_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_59_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_59_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_59_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_59_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_59_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_59_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_59_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_59_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_59_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_59_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_59_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_59_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_59_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_59_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_59_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_59_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_60_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_60_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_60_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_60_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_60_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_60_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_60_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_60_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_60_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_60_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_60_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_60_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_60_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_60_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_60_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_60_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_61_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_61_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_61_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_61_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_61_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_61_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_61_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_61_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_61_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_61_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_61_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_61_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_61_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_61_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_61_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_61_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_62_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_62_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_62_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_62_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_62_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_62_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_62_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_62_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_62_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_62_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_62_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_62_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_62_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_62_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_62_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_62_modules_norm2_parameters_bias_
        )
        l_self_modules_blocks_modules_63_modules_attn_parameters_q_bias_ = (
            L_self_modules_blocks_modules_63_modules_attn_parameters_q_bias_
        )
        l_self_modules_blocks_modules_63_modules_attn_buffers_k_bias_ = (
            L_self_modules_blocks_modules_63_modules_attn_buffers_k_bias_
        )
        l_self_modules_blocks_modules_63_modules_attn_parameters_v_bias_ = (
            L_self_modules_blocks_modules_63_modules_attn_parameters_v_bias_
        )
        l_self_modules_blocks_modules_63_modules_attn_modules_qkv_parameters_weight_ = (
            L_self_modules_blocks_modules_63_modules_attn_modules_qkv_parameters_weight_
        )
        l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_weight_ = L_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_weight_
        l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_bias_ = (
            L_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_bias_
        )
        l_self_modules_blocks_modules_63_modules_norm1_parameters_weight_ = (
            L_self_modules_blocks_modules_63_modules_norm1_parameters_weight_
        )
        l_self_modules_blocks_modules_63_modules_norm1_parameters_bias_ = (
            L_self_modules_blocks_modules_63_modules_norm1_parameters_bias_
        )
        l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_weight_ = (
            L_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_weight_
        )
        l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_bias_ = (
            L_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_bias_
        )
        l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_weight_ = (
            L_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_weight_
        )
        l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_bias_ = (
            L_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_bias_
        )
        l_self_modules_blocks_modules_63_modules_norm2_parameters_weight_ = (
            L_self_modules_blocks_modules_63_modules_norm2_parameters_weight_
        )
        l_self_modules_blocks_modules_63_modules_norm2_parameters_bias_ = (
            L_self_modules_blocks_modules_63_modules_norm2_parameters_bias_
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
        qkv_bias = torch.cat(
            (
                l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_ = None
        qkv = torch._C._nn.linear(
            x_4,
            weight=l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias,
        )
        l_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias
        ) = None
        reshape = qkv.reshape(1, 257, 3, 16, -1)
        qkv = None
        qkv_1 = reshape.permute(2, 0, 3, 1, 4)
        reshape = None
        unbind = qkv_1.unbind(0)
        qkv_1 = None
        q = unbind[0]
        k = unbind[1]
        v = unbind[2]
        unbind = None
        x_5 = torch._C._nn.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0
        )
        q = k = v = None
        transpose_1 = x_5.transpose(1, 2)
        x_5 = None
        x_6 = transpose_1.reshape(1, 257, 1792)
        transpose_1 = None
        x_7 = torch._C._nn.linear(
            x_6,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.dropout(x_7, 0.0, False, False)
        x_7 = None
        x_9 = torch.nn.functional.layer_norm(
            x_8,
            (1792,),
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_8 = (
            l_self_modules_blocks_modules_0_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_0_modules_norm1_parameters_bias_ = None
        x_10 = x_4 + x_9
        x_4 = x_9 = None
        x_11 = torch._C._nn.linear(
            x_10,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_12 = torch._C._nn.gelu(x_11, approximate="none")
        x_11 = None
        x_13 = torch.nn.functional.dropout(x_12, 0.0, False, False)
        x_12 = None
        x_14 = torch._C._nn.linear(
            x_13,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_13 = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_15 = torch.nn.functional.dropout(x_14, 0.0, False, False)
        x_14 = None
        x_16 = torch.nn.functional.layer_norm(
            x_15,
            (1792,),
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_15 = (
            l_self_modules_blocks_modules_0_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_0_modules_norm2_parameters_bias_ = None
        x_17 = x_10 + x_16
        x_10 = x_16 = None
        qkv_bias_1 = torch.cat(
            (
                l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_ = None
        qkv_2 = torch._C._nn.linear(
            x_17,
            weight=l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_1,
        )
        l_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_1
        ) = None
        reshape_2 = qkv_2.reshape(1, 257, 3, 16, -1)
        qkv_2 = None
        qkv_3 = reshape_2.permute(2, 0, 3, 1, 4)
        reshape_2 = None
        unbind_1 = qkv_3.unbind(0)
        qkv_3 = None
        q_1 = unbind_1[0]
        k_1 = unbind_1[1]
        v_1 = unbind_1[2]
        unbind_1 = None
        x_18 = torch._C._nn.scaled_dot_product_attention(
            q_1, k_1, v_1, attn_mask=None, dropout_p=0.0
        )
        q_1 = k_1 = v_1 = None
        transpose_2 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_2.reshape(1, 257, 1792)
        transpose_2 = None
        x_20 = torch._C._nn.linear(
            x_19,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.dropout(x_20, 0.0, False, False)
        x_20 = None
        x_22 = torch.nn.functional.layer_norm(
            x_21,
            (1792,),
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_1_modules_norm1_parameters_bias_ = None
        x_23 = x_17 + x_22
        x_17 = x_22 = None
        x_24 = torch._C._nn.linear(
            x_23,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_25 = torch._C._nn.gelu(x_24, approximate="none")
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.0, False, False)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_26 = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.dropout(x_27, 0.0, False, False)
        x_27 = None
        x_29 = torch.nn.functional.layer_norm(
            x_28,
            (1792,),
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_28 = (
            l_self_modules_blocks_modules_1_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_1_modules_norm2_parameters_bias_ = None
        x_30 = x_23 + x_29
        x_23 = x_29 = None
        qkv_bias_2 = torch.cat(
            (
                l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_ = None
        qkv_4 = torch._C._nn.linear(
            x_30,
            weight=l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_2,
        )
        l_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_2
        ) = None
        reshape_4 = qkv_4.reshape(1, 257, 3, 16, -1)
        qkv_4 = None
        qkv_5 = reshape_4.permute(2, 0, 3, 1, 4)
        reshape_4 = None
        unbind_2 = qkv_5.unbind(0)
        qkv_5 = None
        q_2 = unbind_2[0]
        k_2 = unbind_2[1]
        v_2 = unbind_2[2]
        unbind_2 = None
        x_31 = torch._C._nn.scaled_dot_product_attention(
            q_2, k_2, v_2, attn_mask=None, dropout_p=0.0
        )
        q_2 = k_2 = v_2 = None
        transpose_3 = x_31.transpose(1, 2)
        x_31 = None
        x_32 = transpose_3.reshape(1, 257, 1792)
        transpose_3 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_,
        )
        x_32 = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.dropout(x_33, 0.0, False, False)
        x_33 = None
        x_35 = torch.nn.functional.layer_norm(
            x_34,
            (1792,),
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_34 = (
            l_self_modules_blocks_modules_2_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_2_modules_norm1_parameters_bias_ = None
        x_36 = x_30 + x_35
        x_30 = x_35 = None
        x_37 = torch._C._nn.linear(
            x_36,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_38 = torch._C._nn.gelu(x_37, approximate="none")
        x_37 = None
        x_39 = torch.nn.functional.dropout(x_38, 0.0, False, False)
        x_38 = None
        x_40 = torch._C._nn.linear(
            x_39,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_39 = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.dropout(x_40, 0.0, False, False)
        x_40 = None
        x_42 = torch.nn.functional.layer_norm(
            x_41,
            (1792,),
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_41 = (
            l_self_modules_blocks_modules_2_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_2_modules_norm2_parameters_bias_ = None
        x_43 = x_36 + x_42
        x_36 = x_42 = None
        qkv_bias_3 = torch.cat(
            (
                l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_ = None
        qkv_6 = torch._C._nn.linear(
            x_43,
            weight=l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_3,
        )
        l_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_3
        ) = None
        reshape_6 = qkv_6.reshape(1, 257, 3, 16, -1)
        qkv_6 = None
        qkv_7 = reshape_6.permute(2, 0, 3, 1, 4)
        reshape_6 = None
        unbind_3 = qkv_7.unbind(0)
        qkv_7 = None
        q_3 = unbind_3[0]
        k_3 = unbind_3[1]
        v_3 = unbind_3[2]
        unbind_3 = None
        x_44 = torch._C._nn.scaled_dot_product_attention(
            q_3, k_3, v_3, attn_mask=None, dropout_p=0.0
        )
        q_3 = k_3 = v_3 = None
        transpose_4 = x_44.transpose(1, 2)
        x_44 = None
        x_45 = transpose_4.reshape(1, 257, 1792)
        transpose_4 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_,
        )
        x_45 = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.dropout(x_46, 0.0, False, False)
        x_46 = None
        x_48 = torch.nn.functional.layer_norm(
            x_47,
            (1792,),
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_47 = (
            l_self_modules_blocks_modules_3_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_3_modules_norm1_parameters_bias_ = None
        x_49 = x_43 + x_48
        x_43 = x_48 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_51 = torch._C._nn.gelu(x_50, approximate="none")
        x_50 = None
        x_52 = torch.nn.functional.dropout(x_51, 0.0, False, False)
        x_51 = None
        x_53 = torch._C._nn.linear(
            x_52,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_52 = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_54 = torch.nn.functional.dropout(x_53, 0.0, False, False)
        x_53 = None
        x_55 = torch.nn.functional.layer_norm(
            x_54,
            (1792,),
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_54 = (
            l_self_modules_blocks_modules_3_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_3_modules_norm2_parameters_bias_ = None
        x_56 = x_49 + x_55
        x_49 = x_55 = None
        qkv_bias_4 = torch.cat(
            (
                l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_ = None
        qkv_8 = torch._C._nn.linear(
            x_56,
            weight=l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_4,
        )
        l_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_4
        ) = None
        reshape_8 = qkv_8.reshape(1, 257, 3, 16, -1)
        qkv_8 = None
        qkv_9 = reshape_8.permute(2, 0, 3, 1, 4)
        reshape_8 = None
        unbind_4 = qkv_9.unbind(0)
        qkv_9 = None
        q_4 = unbind_4[0]
        k_4 = unbind_4[1]
        v_4 = unbind_4[2]
        unbind_4 = None
        x_57 = torch._C._nn.scaled_dot_product_attention(
            q_4, k_4, v_4, attn_mask=None, dropout_p=0.0
        )
        q_4 = k_4 = v_4 = None
        transpose_5 = x_57.transpose(1, 2)
        x_57 = None
        x_58 = transpose_5.reshape(1, 257, 1792)
        transpose_5 = None
        x_59 = torch._C._nn.linear(
            x_58,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_,
        )
        x_58 = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_60 = torch.nn.functional.dropout(x_59, 0.0, False, False)
        x_59 = None
        x_61 = torch.nn.functional.layer_norm(
            x_60,
            (1792,),
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_60 = (
            l_self_modules_blocks_modules_4_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_4_modules_norm1_parameters_bias_ = None
        x_62 = x_56 + x_61
        x_56 = x_61 = None
        x_63 = torch._C._nn.linear(
            x_62,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_64 = torch._C._nn.gelu(x_63, approximate="none")
        x_63 = None
        x_65 = torch.nn.functional.dropout(x_64, 0.0, False, False)
        x_64 = None
        x_66 = torch._C._nn.linear(
            x_65,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_65 = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.dropout(x_66, 0.0, False, False)
        x_66 = None
        x_68 = torch.nn.functional.layer_norm(
            x_67,
            (1792,),
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_67 = (
            l_self_modules_blocks_modules_4_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_4_modules_norm2_parameters_bias_ = None
        x_69 = x_62 + x_68
        x_62 = x_68 = None
        qkv_bias_5 = torch.cat(
            (
                l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_ = None
        qkv_10 = torch._C._nn.linear(
            x_69,
            weight=l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_5,
        )
        l_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_5
        ) = None
        reshape_10 = qkv_10.reshape(1, 257, 3, 16, -1)
        qkv_10 = None
        qkv_11 = reshape_10.permute(2, 0, 3, 1, 4)
        reshape_10 = None
        unbind_5 = qkv_11.unbind(0)
        qkv_11 = None
        q_5 = unbind_5[0]
        k_5 = unbind_5[1]
        v_5 = unbind_5[2]
        unbind_5 = None
        x_70 = torch._C._nn.scaled_dot_product_attention(
            q_5, k_5, v_5, attn_mask=None, dropout_p=0.0
        )
        q_5 = k_5 = v_5 = None
        transpose_6 = x_70.transpose(1, 2)
        x_70 = None
        x_71 = transpose_6.reshape(1, 257, 1792)
        transpose_6 = None
        x_72 = torch._C._nn.linear(
            x_71,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_,
        )
        x_71 = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_73 = torch.nn.functional.dropout(x_72, 0.0, False, False)
        x_72 = None
        x_74 = torch.nn.functional.layer_norm(
            x_73,
            (1792,),
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_73 = (
            l_self_modules_blocks_modules_5_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_5_modules_norm1_parameters_bias_ = None
        x_75 = x_69 + x_74
        x_69 = x_74 = None
        x_76 = torch._C._nn.linear(
            x_75,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_77 = torch._C._nn.gelu(x_76, approximate="none")
        x_76 = None
        x_78 = torch.nn.functional.dropout(x_77, 0.0, False, False)
        x_77 = None
        x_79 = torch._C._nn.linear(
            x_78,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_78 = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_80 = torch.nn.functional.dropout(x_79, 0.0, False, False)
        x_79 = None
        x_81 = torch.nn.functional.layer_norm(
            x_80,
            (1792,),
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_80 = (
            l_self_modules_blocks_modules_5_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_5_modules_norm2_parameters_bias_ = None
        x_82 = x_75 + x_81
        x_75 = x_81 = None
        qkv_bias_6 = torch.cat(
            (
                l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_ = None
        qkv_12 = torch._C._nn.linear(
            x_82,
            weight=l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_6,
        )
        l_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_6
        ) = None
        reshape_12 = qkv_12.reshape(1, 257, 3, 16, -1)
        qkv_12 = None
        qkv_13 = reshape_12.permute(2, 0, 3, 1, 4)
        reshape_12 = None
        unbind_6 = qkv_13.unbind(0)
        qkv_13 = None
        q_6 = unbind_6[0]
        k_6 = unbind_6[1]
        v_6 = unbind_6[2]
        unbind_6 = None
        x_83 = torch._C._nn.scaled_dot_product_attention(
            q_6, k_6, v_6, attn_mask=None, dropout_p=0.0
        )
        q_6 = k_6 = v_6 = None
        transpose_7 = x_83.transpose(1, 2)
        x_83 = None
        x_84 = transpose_7.reshape(1, 257, 1792)
        transpose_7 = None
        x_85 = torch._C._nn.linear(
            x_84,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_,
        )
        x_84 = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.dropout(x_85, 0.0, False, False)
        x_85 = None
        x_87 = torch.nn.functional.layer_norm(
            x_86,
            (1792,),
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_86 = (
            l_self_modules_blocks_modules_6_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_6_modules_norm1_parameters_bias_ = None
        x_88 = x_82 + x_87
        x_82 = x_87 = None
        x_89 = torch._C._nn.linear(
            x_88,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_90 = torch._C._nn.gelu(x_89, approximate="none")
        x_89 = None
        x_91 = torch.nn.functional.dropout(x_90, 0.0, False, False)
        x_90 = None
        x_92 = torch._C._nn.linear(
            x_91,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_91 = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.dropout(x_92, 0.0, False, False)
        x_92 = None
        x_94 = torch.nn.functional.layer_norm(
            x_93,
            (1792,),
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_93 = (
            l_self_modules_blocks_modules_6_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_6_modules_norm2_parameters_bias_ = None
        x_95 = x_88 + x_94
        x_88 = x_94 = None
        qkv_bias_7 = torch.cat(
            (
                l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_ = None
        qkv_14 = torch._C._nn.linear(
            x_95,
            weight=l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_7,
        )
        l_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_7
        ) = None
        reshape_14 = qkv_14.reshape(1, 257, 3, 16, -1)
        qkv_14 = None
        qkv_15 = reshape_14.permute(2, 0, 3, 1, 4)
        reshape_14 = None
        unbind_7 = qkv_15.unbind(0)
        qkv_15 = None
        q_7 = unbind_7[0]
        k_7 = unbind_7[1]
        v_7 = unbind_7[2]
        unbind_7 = None
        x_96 = torch._C._nn.scaled_dot_product_attention(
            q_7, k_7, v_7, attn_mask=None, dropout_p=0.0
        )
        q_7 = k_7 = v_7 = None
        transpose_8 = x_96.transpose(1, 2)
        x_96 = None
        x_97 = transpose_8.reshape(1, 257, 1792)
        transpose_8 = None
        x_98 = torch._C._nn.linear(
            x_97,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_,
        )
        x_97 = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.dropout(x_98, 0.0, False, False)
        x_98 = None
        x_100 = torch.nn.functional.layer_norm(
            x_99,
            (1792,),
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_99 = (
            l_self_modules_blocks_modules_7_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_7_modules_norm1_parameters_bias_ = None
        x_101 = x_95 + x_100
        x_95 = x_100 = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_103 = torch._C._nn.gelu(x_102, approximate="none")
        x_102 = None
        x_104 = torch.nn.functional.dropout(x_103, 0.0, False, False)
        x_103 = None
        x_105 = torch._C._nn.linear(
            x_104,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_104 = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.dropout(x_105, 0.0, False, False)
        x_105 = None
        x_107 = torch.nn.functional.layer_norm(
            x_106,
            (1792,),
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_106 = (
            l_self_modules_blocks_modules_7_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_7_modules_norm2_parameters_bias_ = None
        x_108 = x_101 + x_107
        x_101 = x_107 = None
        qkv_bias_8 = torch.cat(
            (
                l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_ = None
        qkv_16 = torch._C._nn.linear(
            x_108,
            weight=l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_8,
        )
        l_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_8
        ) = None
        reshape_16 = qkv_16.reshape(1, 257, 3, 16, -1)
        qkv_16 = None
        qkv_17 = reshape_16.permute(2, 0, 3, 1, 4)
        reshape_16 = None
        unbind_8 = qkv_17.unbind(0)
        qkv_17 = None
        q_8 = unbind_8[0]
        k_8 = unbind_8[1]
        v_8 = unbind_8[2]
        unbind_8 = None
        x_109 = torch._C._nn.scaled_dot_product_attention(
            q_8, k_8, v_8, attn_mask=None, dropout_p=0.0
        )
        q_8 = k_8 = v_8 = None
        transpose_9 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_9.reshape(1, 257, 1792)
        transpose_9 = None
        x_111 = torch._C._nn.linear(
            x_110,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_,
        )
        x_110 = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.dropout(x_111, 0.0, False, False)
        x_111 = None
        x_113 = torch.nn.functional.layer_norm(
            x_112,
            (1792,),
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_112 = (
            l_self_modules_blocks_modules_8_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_8_modules_norm1_parameters_bias_ = None
        x_114 = x_108 + x_113
        x_108 = x_113 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_116 = torch._C._nn.gelu(x_115, approximate="none")
        x_115 = None
        x_117 = torch.nn.functional.dropout(x_116, 0.0, False, False)
        x_116 = None
        x_118 = torch._C._nn.linear(
            x_117,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_117 = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_119 = torch.nn.functional.dropout(x_118, 0.0, False, False)
        x_118 = None
        x_120 = torch.nn.functional.layer_norm(
            x_119,
            (1792,),
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_119 = (
            l_self_modules_blocks_modules_8_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_8_modules_norm2_parameters_bias_ = None
        x_121 = x_114 + x_120
        x_114 = x_120 = None
        qkv_bias_9 = torch.cat(
            (
                l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_ = None
        qkv_18 = torch._C._nn.linear(
            x_121,
            weight=l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_9,
        )
        l_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_9
        ) = None
        reshape_18 = qkv_18.reshape(1, 257, 3, 16, -1)
        qkv_18 = None
        qkv_19 = reshape_18.permute(2, 0, 3, 1, 4)
        reshape_18 = None
        unbind_9 = qkv_19.unbind(0)
        qkv_19 = None
        q_9 = unbind_9[0]
        k_9 = unbind_9[1]
        v_9 = unbind_9[2]
        unbind_9 = None
        x_122 = torch._C._nn.scaled_dot_product_attention(
            q_9, k_9, v_9, attn_mask=None, dropout_p=0.0
        )
        q_9 = k_9 = v_9 = None
        transpose_10 = x_122.transpose(1, 2)
        x_122 = None
        x_123 = transpose_10.reshape(1, 257, 1792)
        transpose_10 = None
        x_124 = torch._C._nn.linear(
            x_123,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_,
        )
        x_123 = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.dropout(x_124, 0.0, False, False)
        x_124 = None
        x_126 = torch.nn.functional.layer_norm(
            x_125,
            (1792,),
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_125 = (
            l_self_modules_blocks_modules_9_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_9_modules_norm1_parameters_bias_ = None
        x_127 = x_121 + x_126
        x_121 = x_126 = None
        x_128 = torch._C._nn.linear(
            x_127,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_129 = torch._C._nn.gelu(x_128, approximate="none")
        x_128 = None
        x_130 = torch.nn.functional.dropout(x_129, 0.0, False, False)
        x_129 = None
        x_131 = torch._C._nn.linear(
            x_130,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_130 = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.dropout(x_131, 0.0, False, False)
        x_131 = None
        x_133 = torch.nn.functional.layer_norm(
            x_132,
            (1792,),
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_132 = (
            l_self_modules_blocks_modules_9_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_9_modules_norm2_parameters_bias_ = None
        x_134 = x_127 + x_133
        x_127 = x_133 = None
        qkv_bias_10 = torch.cat(
            (
                l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_ = None
        qkv_20 = torch._C._nn.linear(
            x_134,
            weight=l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_10,
        )
        l_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_10
        ) = None
        reshape_20 = qkv_20.reshape(1, 257, 3, 16, -1)
        qkv_20 = None
        qkv_21 = reshape_20.permute(2, 0, 3, 1, 4)
        reshape_20 = None
        unbind_10 = qkv_21.unbind(0)
        qkv_21 = None
        q_10 = unbind_10[0]
        k_10 = unbind_10[1]
        v_10 = unbind_10[2]
        unbind_10 = None
        x_135 = torch._C._nn.scaled_dot_product_attention(
            q_10, k_10, v_10, attn_mask=None, dropout_p=0.0
        )
        q_10 = k_10 = v_10 = None
        transpose_11 = x_135.transpose(1, 2)
        x_135 = None
        x_136 = transpose_11.reshape(1, 257, 1792)
        transpose_11 = None
        x_137 = torch._C._nn.linear(
            x_136,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_,
        )
        x_136 = l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.dropout(x_137, 0.0, False, False)
        x_137 = None
        x_139 = torch.nn.functional.layer_norm(
            x_138,
            (1792,),
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_138 = (
            l_self_modules_blocks_modules_10_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_10_modules_norm1_parameters_bias_ = None
        x_140 = x_134 + x_139
        x_134 = x_139 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_142 = torch._C._nn.gelu(x_141, approximate="none")
        x_141 = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_143 = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_145 = torch.nn.functional.dropout(x_144, 0.0, False, False)
        x_144 = None
        x_146 = torch.nn.functional.layer_norm(
            x_145,
            (1792,),
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_145 = (
            l_self_modules_blocks_modules_10_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_10_modules_norm2_parameters_bias_ = None
        x_147 = x_140 + x_146
        x_140 = x_146 = None
        qkv_bias_11 = torch.cat(
            (
                l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_ = None
        qkv_22 = torch._C._nn.linear(
            x_147,
            weight=l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_11,
        )
        l_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_11
        ) = None
        reshape_22 = qkv_22.reshape(1, 257, 3, 16, -1)
        qkv_22 = None
        qkv_23 = reshape_22.permute(2, 0, 3, 1, 4)
        reshape_22 = None
        unbind_11 = qkv_23.unbind(0)
        qkv_23 = None
        q_11 = unbind_11[0]
        k_11 = unbind_11[1]
        v_11 = unbind_11[2]
        unbind_11 = None
        x_148 = torch._C._nn.scaled_dot_product_attention(
            q_11, k_11, v_11, attn_mask=None, dropout_p=0.0
        )
        q_11 = k_11 = v_11 = None
        transpose_12 = x_148.transpose(1, 2)
        x_148 = None
        x_149 = transpose_12.reshape(1, 257, 1792)
        transpose_12 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_,
        )
        x_149 = l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_151 = torch.nn.functional.dropout(x_150, 0.0, False, False)
        x_150 = None
        x_152 = torch.nn.functional.layer_norm(
            x_151,
            (1792,),
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_151 = (
            l_self_modules_blocks_modules_11_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_11_modules_norm1_parameters_bias_ = None
        x_153 = x_147 + x_152
        x_147 = x_152 = None
        x_154 = torch._C._nn.linear(
            x_153,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_155 = torch._C._nn.gelu(x_154, approximate="none")
        x_154 = None
        x_156 = torch.nn.functional.dropout(x_155, 0.0, False, False)
        x_155 = None
        x_157 = torch._C._nn.linear(
            x_156,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_156 = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_158 = torch.nn.functional.dropout(x_157, 0.0, False, False)
        x_157 = None
        x_159 = torch.nn.functional.layer_norm(
            x_158,
            (1792,),
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_158 = (
            l_self_modules_blocks_modules_11_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_11_modules_norm2_parameters_bias_ = None
        x_160 = x_153 + x_159
        x_153 = x_159 = None
        qkv_bias_12 = torch.cat(
            (
                l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_12_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_12_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_12_modules_attn_parameters_v_bias_ = None
        qkv_24 = torch._C._nn.linear(
            x_160,
            weight=l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_12,
        )
        l_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_12
        ) = None
        reshape_24 = qkv_24.reshape(1, 257, 3, 16, -1)
        qkv_24 = None
        qkv_25 = reshape_24.permute(2, 0, 3, 1, 4)
        reshape_24 = None
        unbind_12 = qkv_25.unbind(0)
        qkv_25 = None
        q_12 = unbind_12[0]
        k_12 = unbind_12[1]
        v_12 = unbind_12[2]
        unbind_12 = None
        x_161 = torch._C._nn.scaled_dot_product_attention(
            q_12, k_12, v_12, attn_mask=None, dropout_p=0.0
        )
        q_12 = k_12 = v_12 = None
        transpose_13 = x_161.transpose(1, 2)
        x_161 = None
        x_162 = transpose_13.reshape(1, 257, 1792)
        transpose_13 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_,
        )
        x_162 = l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.dropout(x_163, 0.0, False, False)
        x_163 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (1792,),
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_164 = (
            l_self_modules_blocks_modules_12_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_12_modules_norm1_parameters_bias_ = None
        x_166 = x_160 + x_165
        x_160 = x_165 = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_168 = torch._C._nn.gelu(x_167, approximate="none")
        x_167 = None
        x_169 = torch.nn.functional.dropout(x_168, 0.0, False, False)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_169 = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_171 = torch.nn.functional.dropout(x_170, 0.0, False, False)
        x_170 = None
        x_172 = torch.nn.functional.layer_norm(
            x_171,
            (1792,),
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_171 = (
            l_self_modules_blocks_modules_12_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_12_modules_norm2_parameters_bias_ = None
        x_173 = x_166 + x_172
        x_166 = x_172 = None
        qkv_bias_13 = torch.cat(
            (
                l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_13_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_13_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_13_modules_attn_parameters_v_bias_ = None
        qkv_26 = torch._C._nn.linear(
            x_173,
            weight=l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_13,
        )
        l_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_13
        ) = None
        reshape_26 = qkv_26.reshape(1, 257, 3, 16, -1)
        qkv_26 = None
        qkv_27 = reshape_26.permute(2, 0, 3, 1, 4)
        reshape_26 = None
        unbind_13 = qkv_27.unbind(0)
        qkv_27 = None
        q_13 = unbind_13[0]
        k_13 = unbind_13[1]
        v_13 = unbind_13[2]
        unbind_13 = None
        x_174 = torch._C._nn.scaled_dot_product_attention(
            q_13, k_13, v_13, attn_mask=None, dropout_p=0.0
        )
        q_13 = k_13 = v_13 = None
        transpose_14 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_14.reshape(1, 257, 1792)
        transpose_14 = None
        x_176 = torch._C._nn.linear(
            x_175,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_,
        )
        x_175 = l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_177 = torch.nn.functional.dropout(x_176, 0.0, False, False)
        x_176 = None
        x_178 = torch.nn.functional.layer_norm(
            x_177,
            (1792,),
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_177 = (
            l_self_modules_blocks_modules_13_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_13_modules_norm1_parameters_bias_ = None
        x_179 = x_173 + x_178
        x_173 = x_178 = None
        x_180 = torch._C._nn.linear(
            x_179,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_181 = torch._C._nn.gelu(x_180, approximate="none")
        x_180 = None
        x_182 = torch.nn.functional.dropout(x_181, 0.0, False, False)
        x_181 = None
        x_183 = torch._C._nn.linear(
            x_182,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_182 = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.dropout(x_183, 0.0, False, False)
        x_183 = None
        x_185 = torch.nn.functional.layer_norm(
            x_184,
            (1792,),
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_184 = (
            l_self_modules_blocks_modules_13_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_13_modules_norm2_parameters_bias_ = None
        x_186 = x_179 + x_185
        x_179 = x_185 = None
        qkv_bias_14 = torch.cat(
            (
                l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_14_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_14_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_14_modules_attn_parameters_v_bias_ = None
        qkv_28 = torch._C._nn.linear(
            x_186,
            weight=l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_14,
        )
        l_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_14
        ) = None
        reshape_28 = qkv_28.reshape(1, 257, 3, 16, -1)
        qkv_28 = None
        qkv_29 = reshape_28.permute(2, 0, 3, 1, 4)
        reshape_28 = None
        unbind_14 = qkv_29.unbind(0)
        qkv_29 = None
        q_14 = unbind_14[0]
        k_14 = unbind_14[1]
        v_14 = unbind_14[2]
        unbind_14 = None
        x_187 = torch._C._nn.scaled_dot_product_attention(
            q_14, k_14, v_14, attn_mask=None, dropout_p=0.0
        )
        q_14 = k_14 = v_14 = None
        transpose_15 = x_187.transpose(1, 2)
        x_187 = None
        x_188 = transpose_15.reshape(1, 257, 1792)
        transpose_15 = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_,
        )
        x_188 = l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.dropout(x_189, 0.0, False, False)
        x_189 = None
        x_191 = torch.nn.functional.layer_norm(
            x_190,
            (1792,),
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_190 = (
            l_self_modules_blocks_modules_14_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_14_modules_norm1_parameters_bias_ = None
        x_192 = x_186 + x_191
        x_186 = x_191 = None
        x_193 = torch._C._nn.linear(
            x_192,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_194 = torch._C._nn.gelu(x_193, approximate="none")
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.0, False, False)
        x_194 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_195 = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.dropout(x_196, 0.0, False, False)
        x_196 = None
        x_198 = torch.nn.functional.layer_norm(
            x_197,
            (1792,),
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_197 = (
            l_self_modules_blocks_modules_14_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_14_modules_norm2_parameters_bias_ = None
        x_199 = x_192 + x_198
        x_192 = x_198 = None
        qkv_bias_15 = torch.cat(
            (
                l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_15_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_15_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_15_modules_attn_parameters_v_bias_ = None
        qkv_30 = torch._C._nn.linear(
            x_199,
            weight=l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_15,
        )
        l_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_15
        ) = None
        reshape_30 = qkv_30.reshape(1, 257, 3, 16, -1)
        qkv_30 = None
        qkv_31 = reshape_30.permute(2, 0, 3, 1, 4)
        reshape_30 = None
        unbind_15 = qkv_31.unbind(0)
        qkv_31 = None
        q_15 = unbind_15[0]
        k_15 = unbind_15[1]
        v_15 = unbind_15[2]
        unbind_15 = None
        x_200 = torch._C._nn.scaled_dot_product_attention(
            q_15, k_15, v_15, attn_mask=None, dropout_p=0.0
        )
        q_15 = k_15 = v_15 = None
        transpose_16 = x_200.transpose(1, 2)
        x_200 = None
        x_201 = transpose_16.reshape(1, 257, 1792)
        transpose_16 = None
        x_202 = torch._C._nn.linear(
            x_201,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_,
        )
        x_201 = l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_203 = torch.nn.functional.dropout(x_202, 0.0, False, False)
        x_202 = None
        x_204 = torch.nn.functional.layer_norm(
            x_203,
            (1792,),
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_203 = (
            l_self_modules_blocks_modules_15_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_15_modules_norm1_parameters_bias_ = None
        x_205 = x_199 + x_204
        x_199 = x_204 = None
        x_206 = torch._C._nn.linear(
            x_205,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_207 = torch._C._nn.gelu(x_206, approximate="none")
        x_206 = None
        x_208 = torch.nn.functional.dropout(x_207, 0.0, False, False)
        x_207 = None
        x_209 = torch._C._nn.linear(
            x_208,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_208 = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_210 = torch.nn.functional.dropout(x_209, 0.0, False, False)
        x_209 = None
        x_211 = torch.nn.functional.layer_norm(
            x_210,
            (1792,),
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_210 = (
            l_self_modules_blocks_modules_15_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_15_modules_norm2_parameters_bias_ = None
        x_212 = x_205 + x_211
        x_205 = x_211 = None
        qkv_bias_16 = torch.cat(
            (
                l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_16_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_16_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_16_modules_attn_parameters_v_bias_ = None
        qkv_32 = torch._C._nn.linear(
            x_212,
            weight=l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_16,
        )
        l_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_16
        ) = None
        reshape_32 = qkv_32.reshape(1, 257, 3, 16, -1)
        qkv_32 = None
        qkv_33 = reshape_32.permute(2, 0, 3, 1, 4)
        reshape_32 = None
        unbind_16 = qkv_33.unbind(0)
        qkv_33 = None
        q_16 = unbind_16[0]
        k_16 = unbind_16[1]
        v_16 = unbind_16[2]
        unbind_16 = None
        x_213 = torch._C._nn.scaled_dot_product_attention(
            q_16, k_16, v_16, attn_mask=None, dropout_p=0.0
        )
        q_16 = k_16 = v_16 = None
        transpose_17 = x_213.transpose(1, 2)
        x_213 = None
        x_214 = transpose_17.reshape(1, 257, 1792)
        transpose_17 = None
        x_215 = torch._C._nn.linear(
            x_214,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_,
        )
        x_214 = l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.dropout(x_215, 0.0, False, False)
        x_215 = None
        x_217 = torch.nn.functional.layer_norm(
            x_216,
            (1792,),
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_216 = (
            l_self_modules_blocks_modules_16_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_16_modules_norm1_parameters_bias_ = None
        x_218 = x_212 + x_217
        x_212 = x_217 = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_220 = torch._C._nn.gelu(x_219, approximate="none")
        x_219 = None
        x_221 = torch.nn.functional.dropout(x_220, 0.0, False, False)
        x_220 = None
        x_222 = torch._C._nn.linear(
            x_221,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_221 = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.dropout(x_222, 0.0, False, False)
        x_222 = None
        x_224 = torch.nn.functional.layer_norm(
            x_223,
            (1792,),
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_223 = (
            l_self_modules_blocks_modules_16_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_16_modules_norm2_parameters_bias_ = None
        x_225 = x_218 + x_224
        x_218 = x_224 = None
        qkv_bias_17 = torch.cat(
            (
                l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_17_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_17_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_17_modules_attn_parameters_v_bias_ = None
        qkv_34 = torch._C._nn.linear(
            x_225,
            weight=l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_17,
        )
        l_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_17
        ) = None
        reshape_34 = qkv_34.reshape(1, 257, 3, 16, -1)
        qkv_34 = None
        qkv_35 = reshape_34.permute(2, 0, 3, 1, 4)
        reshape_34 = None
        unbind_17 = qkv_35.unbind(0)
        qkv_35 = None
        q_17 = unbind_17[0]
        k_17 = unbind_17[1]
        v_17 = unbind_17[2]
        unbind_17 = None
        x_226 = torch._C._nn.scaled_dot_product_attention(
            q_17, k_17, v_17, attn_mask=None, dropout_p=0.0
        )
        q_17 = k_17 = v_17 = None
        transpose_18 = x_226.transpose(1, 2)
        x_226 = None
        x_227 = transpose_18.reshape(1, 257, 1792)
        transpose_18 = None
        x_228 = torch._C._nn.linear(
            x_227,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_,
        )
        x_227 = l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_229 = torch.nn.functional.dropout(x_228, 0.0, False, False)
        x_228 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (1792,),
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_229 = (
            l_self_modules_blocks_modules_17_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_17_modules_norm1_parameters_bias_ = None
        x_231 = x_225 + x_230
        x_225 = x_230 = None
        x_232 = torch._C._nn.linear(
            x_231,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_233 = torch._C._nn.gelu(x_232, approximate="none")
        x_232 = None
        x_234 = torch.nn.functional.dropout(x_233, 0.0, False, False)
        x_233 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_234 = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_236 = torch.nn.functional.dropout(x_235, 0.0, False, False)
        x_235 = None
        x_237 = torch.nn.functional.layer_norm(
            x_236,
            (1792,),
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_236 = (
            l_self_modules_blocks_modules_17_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_17_modules_norm2_parameters_bias_ = None
        x_238 = x_231 + x_237
        x_231 = x_237 = None
        qkv_bias_18 = torch.cat(
            (
                l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_18_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_18_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_18_modules_attn_parameters_v_bias_ = None
        qkv_36 = torch._C._nn.linear(
            x_238,
            weight=l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_18,
        )
        l_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_18
        ) = None
        reshape_36 = qkv_36.reshape(1, 257, 3, 16, -1)
        qkv_36 = None
        qkv_37 = reshape_36.permute(2, 0, 3, 1, 4)
        reshape_36 = None
        unbind_18 = qkv_37.unbind(0)
        qkv_37 = None
        q_18 = unbind_18[0]
        k_18 = unbind_18[1]
        v_18 = unbind_18[2]
        unbind_18 = None
        x_239 = torch._C._nn.scaled_dot_product_attention(
            q_18, k_18, v_18, attn_mask=None, dropout_p=0.0
        )
        q_18 = k_18 = v_18 = None
        transpose_19 = x_239.transpose(1, 2)
        x_239 = None
        x_240 = transpose_19.reshape(1, 257, 1792)
        transpose_19 = None
        x_241 = torch._C._nn.linear(
            x_240,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_,
        )
        x_240 = l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_242 = torch.nn.functional.dropout(x_241, 0.0, False, False)
        x_241 = None
        x_243 = torch.nn.functional.layer_norm(
            x_242,
            (1792,),
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_242 = (
            l_self_modules_blocks_modules_18_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_18_modules_norm1_parameters_bias_ = None
        x_244 = x_238 + x_243
        x_238 = x_243 = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_246 = torch._C._nn.gelu(x_245, approximate="none")
        x_245 = None
        x_247 = torch.nn.functional.dropout(x_246, 0.0, False, False)
        x_246 = None
        x_248 = torch._C._nn.linear(
            x_247,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_247 = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_249 = torch.nn.functional.dropout(x_248, 0.0, False, False)
        x_248 = None
        x_250 = torch.nn.functional.layer_norm(
            x_249,
            (1792,),
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_249 = (
            l_self_modules_blocks_modules_18_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_18_modules_norm2_parameters_bias_ = None
        x_251 = x_244 + x_250
        x_244 = x_250 = None
        qkv_bias_19 = torch.cat(
            (
                l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_19_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_19_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_19_modules_attn_parameters_v_bias_ = None
        qkv_38 = torch._C._nn.linear(
            x_251,
            weight=l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_19,
        )
        l_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_19
        ) = None
        reshape_38 = qkv_38.reshape(1, 257, 3, 16, -1)
        qkv_38 = None
        qkv_39 = reshape_38.permute(2, 0, 3, 1, 4)
        reshape_38 = None
        unbind_19 = qkv_39.unbind(0)
        qkv_39 = None
        q_19 = unbind_19[0]
        k_19 = unbind_19[1]
        v_19 = unbind_19[2]
        unbind_19 = None
        x_252 = torch._C._nn.scaled_dot_product_attention(
            q_19, k_19, v_19, attn_mask=None, dropout_p=0.0
        )
        q_19 = k_19 = v_19 = None
        transpose_20 = x_252.transpose(1, 2)
        x_252 = None
        x_253 = transpose_20.reshape(1, 257, 1792)
        transpose_20 = None
        x_254 = torch._C._nn.linear(
            x_253,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_,
        )
        x_253 = l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        x_256 = torch.nn.functional.layer_norm(
            x_255,
            (1792,),
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_255 = (
            l_self_modules_blocks_modules_19_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_19_modules_norm1_parameters_bias_ = None
        x_257 = x_251 + x_256
        x_251 = x_256 = None
        x_258 = torch._C._nn.linear(
            x_257,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_259 = torch._C._nn.gelu(x_258, approximate="none")
        x_258 = None
        x_260 = torch.nn.functional.dropout(x_259, 0.0, False, False)
        x_259 = None
        x_261 = torch._C._nn.linear(
            x_260,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_260 = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_262 = torch.nn.functional.dropout(x_261, 0.0, False, False)
        x_261 = None
        x_263 = torch.nn.functional.layer_norm(
            x_262,
            (1792,),
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_262 = (
            l_self_modules_blocks_modules_19_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_19_modules_norm2_parameters_bias_ = None
        x_264 = x_257 + x_263
        x_257 = x_263 = None
        qkv_bias_20 = torch.cat(
            (
                l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_20_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_20_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_20_modules_attn_parameters_v_bias_ = None
        qkv_40 = torch._C._nn.linear(
            x_264,
            weight=l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_20,
        )
        l_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_20
        ) = None
        reshape_40 = qkv_40.reshape(1, 257, 3, 16, -1)
        qkv_40 = None
        qkv_41 = reshape_40.permute(2, 0, 3, 1, 4)
        reshape_40 = None
        unbind_20 = qkv_41.unbind(0)
        qkv_41 = None
        q_20 = unbind_20[0]
        k_20 = unbind_20[1]
        v_20 = unbind_20[2]
        unbind_20 = None
        x_265 = torch._C._nn.scaled_dot_product_attention(
            q_20, k_20, v_20, attn_mask=None, dropout_p=0.0
        )
        q_20 = k_20 = v_20 = None
        transpose_21 = x_265.transpose(1, 2)
        x_265 = None
        x_266 = transpose_21.reshape(1, 257, 1792)
        transpose_21 = None
        x_267 = torch._C._nn.linear(
            x_266,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_,
        )
        x_266 = l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_268 = torch.nn.functional.dropout(x_267, 0.0, False, False)
        x_267 = None
        x_269 = torch.nn.functional.layer_norm(
            x_268,
            (1792,),
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_268 = (
            l_self_modules_blocks_modules_20_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_20_modules_norm1_parameters_bias_ = None
        x_270 = x_264 + x_269
        x_264 = x_269 = None
        x_271 = torch._C._nn.linear(
            x_270,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_272 = torch._C._nn.gelu(x_271, approximate="none")
        x_271 = None
        x_273 = torch.nn.functional.dropout(x_272, 0.0, False, False)
        x_272 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_273 = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_275 = torch.nn.functional.dropout(x_274, 0.0, False, False)
        x_274 = None
        x_276 = torch.nn.functional.layer_norm(
            x_275,
            (1792,),
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_275 = (
            l_self_modules_blocks_modules_20_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_20_modules_norm2_parameters_bias_ = None
        x_277 = x_270 + x_276
        x_270 = x_276 = None
        qkv_bias_21 = torch.cat(
            (
                l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_21_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_21_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_21_modules_attn_parameters_v_bias_ = None
        qkv_42 = torch._C._nn.linear(
            x_277,
            weight=l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_21,
        )
        l_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_21
        ) = None
        reshape_42 = qkv_42.reshape(1, 257, 3, 16, -1)
        qkv_42 = None
        qkv_43 = reshape_42.permute(2, 0, 3, 1, 4)
        reshape_42 = None
        unbind_21 = qkv_43.unbind(0)
        qkv_43 = None
        q_21 = unbind_21[0]
        k_21 = unbind_21[1]
        v_21 = unbind_21[2]
        unbind_21 = None
        x_278 = torch._C._nn.scaled_dot_product_attention(
            q_21, k_21, v_21, attn_mask=None, dropout_p=0.0
        )
        q_21 = k_21 = v_21 = None
        transpose_22 = x_278.transpose(1, 2)
        x_278 = None
        x_279 = transpose_22.reshape(1, 257, 1792)
        transpose_22 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_,
        )
        x_279 = l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_281 = torch.nn.functional.dropout(x_280, 0.0, False, False)
        x_280 = None
        x_282 = torch.nn.functional.layer_norm(
            x_281,
            (1792,),
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_281 = (
            l_self_modules_blocks_modules_21_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_21_modules_norm1_parameters_bias_ = None
        x_283 = x_277 + x_282
        x_277 = x_282 = None
        x_284 = torch._C._nn.linear(
            x_283,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_285 = torch._C._nn.gelu(x_284, approximate="none")
        x_284 = None
        x_286 = torch.nn.functional.dropout(x_285, 0.0, False, False)
        x_285 = None
        x_287 = torch._C._nn.linear(
            x_286,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_286 = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_288 = torch.nn.functional.dropout(x_287, 0.0, False, False)
        x_287 = None
        x_289 = torch.nn.functional.layer_norm(
            x_288,
            (1792,),
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_288 = (
            l_self_modules_blocks_modules_21_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_21_modules_norm2_parameters_bias_ = None
        x_290 = x_283 + x_289
        x_283 = x_289 = None
        qkv_bias_22 = torch.cat(
            (
                l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_22_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_22_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_22_modules_attn_parameters_v_bias_ = None
        qkv_44 = torch._C._nn.linear(
            x_290,
            weight=l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_22,
        )
        l_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_22
        ) = None
        reshape_44 = qkv_44.reshape(1, 257, 3, 16, -1)
        qkv_44 = None
        qkv_45 = reshape_44.permute(2, 0, 3, 1, 4)
        reshape_44 = None
        unbind_22 = qkv_45.unbind(0)
        qkv_45 = None
        q_22 = unbind_22[0]
        k_22 = unbind_22[1]
        v_22 = unbind_22[2]
        unbind_22 = None
        x_291 = torch._C._nn.scaled_dot_product_attention(
            q_22, k_22, v_22, attn_mask=None, dropout_p=0.0
        )
        q_22 = k_22 = v_22 = None
        transpose_23 = x_291.transpose(1, 2)
        x_291 = None
        x_292 = transpose_23.reshape(1, 257, 1792)
        transpose_23 = None
        x_293 = torch._C._nn.linear(
            x_292,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_,
        )
        x_292 = l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_294 = torch.nn.functional.dropout(x_293, 0.0, False, False)
        x_293 = None
        x_295 = torch.nn.functional.layer_norm(
            x_294,
            (1792,),
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_294 = (
            l_self_modules_blocks_modules_22_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_22_modules_norm1_parameters_bias_ = None
        x_296 = x_290 + x_295
        x_290 = x_295 = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_298 = torch._C._nn.gelu(x_297, approximate="none")
        x_297 = None
        x_299 = torch.nn.functional.dropout(x_298, 0.0, False, False)
        x_298 = None
        x_300 = torch._C._nn.linear(
            x_299,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_299 = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_301 = torch.nn.functional.dropout(x_300, 0.0, False, False)
        x_300 = None
        x_302 = torch.nn.functional.layer_norm(
            x_301,
            (1792,),
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_301 = (
            l_self_modules_blocks_modules_22_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_22_modules_norm2_parameters_bias_ = None
        x_303 = x_296 + x_302
        x_296 = x_302 = None
        qkv_bias_23 = torch.cat(
            (
                l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_23_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_23_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_23_modules_attn_parameters_v_bias_ = None
        qkv_46 = torch._C._nn.linear(
            x_303,
            weight=l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_23,
        )
        l_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_23
        ) = None
        reshape_46 = qkv_46.reshape(1, 257, 3, 16, -1)
        qkv_46 = None
        qkv_47 = reshape_46.permute(2, 0, 3, 1, 4)
        reshape_46 = None
        unbind_23 = qkv_47.unbind(0)
        qkv_47 = None
        q_23 = unbind_23[0]
        k_23 = unbind_23[1]
        v_23 = unbind_23[2]
        unbind_23 = None
        x_304 = torch._C._nn.scaled_dot_product_attention(
            q_23, k_23, v_23, attn_mask=None, dropout_p=0.0
        )
        q_23 = k_23 = v_23 = None
        transpose_24 = x_304.transpose(1, 2)
        x_304 = None
        x_305 = transpose_24.reshape(1, 257, 1792)
        transpose_24 = None
        x_306 = torch._C._nn.linear(
            x_305,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_,
        )
        x_305 = l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_307 = torch.nn.functional.dropout(x_306, 0.0, False, False)
        x_306 = None
        x_308 = torch.nn.functional.layer_norm(
            x_307,
            (1792,),
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_307 = (
            l_self_modules_blocks_modules_23_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_23_modules_norm1_parameters_bias_ = None
        x_309 = x_303 + x_308
        x_303 = x_308 = None
        x_310 = torch._C._nn.linear(
            x_309,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_311 = torch._C._nn.gelu(x_310, approximate="none")
        x_310 = None
        x_312 = torch.nn.functional.dropout(x_311, 0.0, False, False)
        x_311 = None
        x_313 = torch._C._nn.linear(
            x_312,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_312 = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_314 = torch.nn.functional.dropout(x_313, 0.0, False, False)
        x_313 = None
        x_315 = torch.nn.functional.layer_norm(
            x_314,
            (1792,),
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_314 = (
            l_self_modules_blocks_modules_23_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_23_modules_norm2_parameters_bias_ = None
        x_316 = x_309 + x_315
        x_309 = x_315 = None
        qkv_bias_24 = torch.cat(
            (
                l_self_modules_blocks_modules_24_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_24_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_24_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_24_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_24_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_24_modules_attn_parameters_v_bias_ = None
        qkv_48 = torch._C._nn.linear(
            x_316,
            weight=l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_24,
        )
        l_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_24
        ) = None
        reshape_48 = qkv_48.reshape(1, 257, 3, 16, -1)
        qkv_48 = None
        qkv_49 = reshape_48.permute(2, 0, 3, 1, 4)
        reshape_48 = None
        unbind_24 = qkv_49.unbind(0)
        qkv_49 = None
        q_24 = unbind_24[0]
        k_24 = unbind_24[1]
        v_24 = unbind_24[2]
        unbind_24 = None
        x_317 = torch._C._nn.scaled_dot_product_attention(
            q_24, k_24, v_24, attn_mask=None, dropout_p=0.0
        )
        q_24 = k_24 = v_24 = None
        transpose_25 = x_317.transpose(1, 2)
        x_317 = None
        x_318 = transpose_25.reshape(1, 257, 1792)
        transpose_25 = None
        x_319 = torch._C._nn.linear(
            x_318,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_,
        )
        x_318 = l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_320 = torch.nn.functional.dropout(x_319, 0.0, False, False)
        x_319 = None
        x_321 = torch.nn.functional.layer_norm(
            x_320,
            (1792,),
            l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_320 = (
            l_self_modules_blocks_modules_24_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_24_modules_norm1_parameters_bias_ = None
        x_322 = x_316 + x_321
        x_316 = x_321 = None
        x_323 = torch._C._nn.linear(
            x_322,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_324 = torch._C._nn.gelu(x_323, approximate="none")
        x_323 = None
        x_325 = torch.nn.functional.dropout(x_324, 0.0, False, False)
        x_324 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_325 = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_327 = torch.nn.functional.dropout(x_326, 0.0, False, False)
        x_326 = None
        x_328 = torch.nn.functional.layer_norm(
            x_327,
            (1792,),
            l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_327 = (
            l_self_modules_blocks_modules_24_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_24_modules_norm2_parameters_bias_ = None
        x_329 = x_322 + x_328
        x_322 = x_328 = None
        qkv_bias_25 = torch.cat(
            (
                l_self_modules_blocks_modules_25_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_25_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_25_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_25_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_25_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_25_modules_attn_parameters_v_bias_ = None
        qkv_50 = torch._C._nn.linear(
            x_329,
            weight=l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_25,
        )
        l_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_25
        ) = None
        reshape_50 = qkv_50.reshape(1, 257, 3, 16, -1)
        qkv_50 = None
        qkv_51 = reshape_50.permute(2, 0, 3, 1, 4)
        reshape_50 = None
        unbind_25 = qkv_51.unbind(0)
        qkv_51 = None
        q_25 = unbind_25[0]
        k_25 = unbind_25[1]
        v_25 = unbind_25[2]
        unbind_25 = None
        x_330 = torch._C._nn.scaled_dot_product_attention(
            q_25, k_25, v_25, attn_mask=None, dropout_p=0.0
        )
        q_25 = k_25 = v_25 = None
        transpose_26 = x_330.transpose(1, 2)
        x_330 = None
        x_331 = transpose_26.reshape(1, 257, 1792)
        transpose_26 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_,
        )
        x_331 = l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_333 = torch.nn.functional.dropout(x_332, 0.0, False, False)
        x_332 = None
        x_334 = torch.nn.functional.layer_norm(
            x_333,
            (1792,),
            l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_333 = (
            l_self_modules_blocks_modules_25_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_25_modules_norm1_parameters_bias_ = None
        x_335 = x_329 + x_334
        x_329 = x_334 = None
        x_336 = torch._C._nn.linear(
            x_335,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_337 = torch._C._nn.gelu(x_336, approximate="none")
        x_336 = None
        x_338 = torch.nn.functional.dropout(x_337, 0.0, False, False)
        x_337 = None
        x_339 = torch._C._nn.linear(
            x_338,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_338 = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_340 = torch.nn.functional.dropout(x_339, 0.0, False, False)
        x_339 = None
        x_341 = torch.nn.functional.layer_norm(
            x_340,
            (1792,),
            l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_340 = (
            l_self_modules_blocks_modules_25_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_25_modules_norm2_parameters_bias_ = None
        x_342 = x_335 + x_341
        x_335 = x_341 = None
        qkv_bias_26 = torch.cat(
            (
                l_self_modules_blocks_modules_26_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_26_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_26_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_26_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_26_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_26_modules_attn_parameters_v_bias_ = None
        qkv_52 = torch._C._nn.linear(
            x_342,
            weight=l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_26,
        )
        l_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_26
        ) = None
        reshape_52 = qkv_52.reshape(1, 257, 3, 16, -1)
        qkv_52 = None
        qkv_53 = reshape_52.permute(2, 0, 3, 1, 4)
        reshape_52 = None
        unbind_26 = qkv_53.unbind(0)
        qkv_53 = None
        q_26 = unbind_26[0]
        k_26 = unbind_26[1]
        v_26 = unbind_26[2]
        unbind_26 = None
        x_343 = torch._C._nn.scaled_dot_product_attention(
            q_26, k_26, v_26, attn_mask=None, dropout_p=0.0
        )
        q_26 = k_26 = v_26 = None
        transpose_27 = x_343.transpose(1, 2)
        x_343 = None
        x_344 = transpose_27.reshape(1, 257, 1792)
        transpose_27 = None
        x_345 = torch._C._nn.linear(
            x_344,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_,
        )
        x_344 = l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_346 = torch.nn.functional.dropout(x_345, 0.0, False, False)
        x_345 = None
        x_347 = torch.nn.functional.layer_norm(
            x_346,
            (1792,),
            l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_346 = (
            l_self_modules_blocks_modules_26_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_26_modules_norm1_parameters_bias_ = None
        x_348 = x_342 + x_347
        x_342 = x_347 = None
        x_349 = torch._C._nn.linear(
            x_348,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_350 = torch._C._nn.gelu(x_349, approximate="none")
        x_349 = None
        x_351 = torch.nn.functional.dropout(x_350, 0.0, False, False)
        x_350 = None
        x_352 = torch._C._nn.linear(
            x_351,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_351 = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_353 = torch.nn.functional.dropout(x_352, 0.0, False, False)
        x_352 = None
        x_354 = torch.nn.functional.layer_norm(
            x_353,
            (1792,),
            l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_353 = (
            l_self_modules_blocks_modules_26_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_26_modules_norm2_parameters_bias_ = None
        x_355 = x_348 + x_354
        x_348 = x_354 = None
        qkv_bias_27 = torch.cat(
            (
                l_self_modules_blocks_modules_27_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_27_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_27_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_27_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_27_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_27_modules_attn_parameters_v_bias_ = None
        qkv_54 = torch._C._nn.linear(
            x_355,
            weight=l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_27,
        )
        l_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_27
        ) = None
        reshape_54 = qkv_54.reshape(1, 257, 3, 16, -1)
        qkv_54 = None
        qkv_55 = reshape_54.permute(2, 0, 3, 1, 4)
        reshape_54 = None
        unbind_27 = qkv_55.unbind(0)
        qkv_55 = None
        q_27 = unbind_27[0]
        k_27 = unbind_27[1]
        v_27 = unbind_27[2]
        unbind_27 = None
        x_356 = torch._C._nn.scaled_dot_product_attention(
            q_27, k_27, v_27, attn_mask=None, dropout_p=0.0
        )
        q_27 = k_27 = v_27 = None
        transpose_28 = x_356.transpose(1, 2)
        x_356 = None
        x_357 = transpose_28.reshape(1, 257, 1792)
        transpose_28 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_,
        )
        x_357 = l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_359 = torch.nn.functional.dropout(x_358, 0.0, False, False)
        x_358 = None
        x_360 = torch.nn.functional.layer_norm(
            x_359,
            (1792,),
            l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_359 = (
            l_self_modules_blocks_modules_27_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_27_modules_norm1_parameters_bias_ = None
        x_361 = x_355 + x_360
        x_355 = x_360 = None
        x_362 = torch._C._nn.linear(
            x_361,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_363 = torch._C._nn.gelu(x_362, approximate="none")
        x_362 = None
        x_364 = torch.nn.functional.dropout(x_363, 0.0, False, False)
        x_363 = None
        x_365 = torch._C._nn.linear(
            x_364,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_364 = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_366 = torch.nn.functional.dropout(x_365, 0.0, False, False)
        x_365 = None
        x_367 = torch.nn.functional.layer_norm(
            x_366,
            (1792,),
            l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_366 = (
            l_self_modules_blocks_modules_27_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_27_modules_norm2_parameters_bias_ = None
        x_368 = x_361 + x_367
        x_361 = x_367 = None
        qkv_bias_28 = torch.cat(
            (
                l_self_modules_blocks_modules_28_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_28_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_28_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_28_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_28_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_28_modules_attn_parameters_v_bias_ = None
        qkv_56 = torch._C._nn.linear(
            x_368,
            weight=l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_28,
        )
        l_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_28
        ) = None
        reshape_56 = qkv_56.reshape(1, 257, 3, 16, -1)
        qkv_56 = None
        qkv_57 = reshape_56.permute(2, 0, 3, 1, 4)
        reshape_56 = None
        unbind_28 = qkv_57.unbind(0)
        qkv_57 = None
        q_28 = unbind_28[0]
        k_28 = unbind_28[1]
        v_28 = unbind_28[2]
        unbind_28 = None
        x_369 = torch._C._nn.scaled_dot_product_attention(
            q_28, k_28, v_28, attn_mask=None, dropout_p=0.0
        )
        q_28 = k_28 = v_28 = None
        transpose_29 = x_369.transpose(1, 2)
        x_369 = None
        x_370 = transpose_29.reshape(1, 257, 1792)
        transpose_29 = None
        x_371 = torch._C._nn.linear(
            x_370,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_,
        )
        x_370 = l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_372 = torch.nn.functional.dropout(x_371, 0.0, False, False)
        x_371 = None
        x_373 = torch.nn.functional.layer_norm(
            x_372,
            (1792,),
            l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_372 = (
            l_self_modules_blocks_modules_28_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_28_modules_norm1_parameters_bias_ = None
        x_374 = x_368 + x_373
        x_368 = x_373 = None
        x_375 = torch._C._nn.linear(
            x_374,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_376 = torch._C._nn.gelu(x_375, approximate="none")
        x_375 = None
        x_377 = torch.nn.functional.dropout(x_376, 0.0, False, False)
        x_376 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_377 = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_379 = torch.nn.functional.dropout(x_378, 0.0, False, False)
        x_378 = None
        x_380 = torch.nn.functional.layer_norm(
            x_379,
            (1792,),
            l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_379 = (
            l_self_modules_blocks_modules_28_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_28_modules_norm2_parameters_bias_ = None
        x_381 = x_374 + x_380
        x_374 = x_380 = None
        qkv_bias_29 = torch.cat(
            (
                l_self_modules_blocks_modules_29_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_29_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_29_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_29_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_29_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_29_modules_attn_parameters_v_bias_ = None
        qkv_58 = torch._C._nn.linear(
            x_381,
            weight=l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_29,
        )
        l_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_29
        ) = None
        reshape_58 = qkv_58.reshape(1, 257, 3, 16, -1)
        qkv_58 = None
        qkv_59 = reshape_58.permute(2, 0, 3, 1, 4)
        reshape_58 = None
        unbind_29 = qkv_59.unbind(0)
        qkv_59 = None
        q_29 = unbind_29[0]
        k_29 = unbind_29[1]
        v_29 = unbind_29[2]
        unbind_29 = None
        x_382 = torch._C._nn.scaled_dot_product_attention(
            q_29, k_29, v_29, attn_mask=None, dropout_p=0.0
        )
        q_29 = k_29 = v_29 = None
        transpose_30 = x_382.transpose(1, 2)
        x_382 = None
        x_383 = transpose_30.reshape(1, 257, 1792)
        transpose_30 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_,
        )
        x_383 = l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_385 = torch.nn.functional.dropout(x_384, 0.0, False, False)
        x_384 = None
        x_386 = torch.nn.functional.layer_norm(
            x_385,
            (1792,),
            l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_385 = (
            l_self_modules_blocks_modules_29_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_29_modules_norm1_parameters_bias_ = None
        x_387 = x_381 + x_386
        x_381 = x_386 = None
        x_388 = torch._C._nn.linear(
            x_387,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_389 = torch._C._nn.gelu(x_388, approximate="none")
        x_388 = None
        x_390 = torch.nn.functional.dropout(x_389, 0.0, False, False)
        x_389 = None
        x_391 = torch._C._nn.linear(
            x_390,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_390 = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_392 = torch.nn.functional.dropout(x_391, 0.0, False, False)
        x_391 = None
        x_393 = torch.nn.functional.layer_norm(
            x_392,
            (1792,),
            l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_392 = (
            l_self_modules_blocks_modules_29_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_29_modules_norm2_parameters_bias_ = None
        x_394 = x_387 + x_393
        x_387 = x_393 = None
        qkv_bias_30 = torch.cat(
            (
                l_self_modules_blocks_modules_30_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_30_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_30_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_30_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_30_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_30_modules_attn_parameters_v_bias_ = None
        qkv_60 = torch._C._nn.linear(
            x_394,
            weight=l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_30,
        )
        l_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_30
        ) = None
        reshape_60 = qkv_60.reshape(1, 257, 3, 16, -1)
        qkv_60 = None
        qkv_61 = reshape_60.permute(2, 0, 3, 1, 4)
        reshape_60 = None
        unbind_30 = qkv_61.unbind(0)
        qkv_61 = None
        q_30 = unbind_30[0]
        k_30 = unbind_30[1]
        v_30 = unbind_30[2]
        unbind_30 = None
        x_395 = torch._C._nn.scaled_dot_product_attention(
            q_30, k_30, v_30, attn_mask=None, dropout_p=0.0
        )
        q_30 = k_30 = v_30 = None
        transpose_31 = x_395.transpose(1, 2)
        x_395 = None
        x_396 = transpose_31.reshape(1, 257, 1792)
        transpose_31 = None
        x_397 = torch._C._nn.linear(
            x_396,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_,
        )
        x_396 = l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_398 = torch.nn.functional.dropout(x_397, 0.0, False, False)
        x_397 = None
        x_399 = torch.nn.functional.layer_norm(
            x_398,
            (1792,),
            l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_398 = (
            l_self_modules_blocks_modules_30_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_30_modules_norm1_parameters_bias_ = None
        x_400 = x_394 + x_399
        x_394 = x_399 = None
        x_401 = torch._C._nn.linear(
            x_400,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_402 = torch._C._nn.gelu(x_401, approximate="none")
        x_401 = None
        x_403 = torch.nn.functional.dropout(x_402, 0.0, False, False)
        x_402 = None
        x_404 = torch._C._nn.linear(
            x_403,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_403 = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_405 = torch.nn.functional.dropout(x_404, 0.0, False, False)
        x_404 = None
        x_406 = torch.nn.functional.layer_norm(
            x_405,
            (1792,),
            l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_405 = (
            l_self_modules_blocks_modules_30_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_30_modules_norm2_parameters_bias_ = None
        x_407 = x_400 + x_406
        x_400 = x_406 = None
        qkv_bias_31 = torch.cat(
            (
                l_self_modules_blocks_modules_31_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_31_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_31_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_31_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_31_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_31_modules_attn_parameters_v_bias_ = None
        qkv_62 = torch._C._nn.linear(
            x_407,
            weight=l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_31,
        )
        l_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_31
        ) = None
        reshape_62 = qkv_62.reshape(1, 257, 3, 16, -1)
        qkv_62 = None
        qkv_63 = reshape_62.permute(2, 0, 3, 1, 4)
        reshape_62 = None
        unbind_31 = qkv_63.unbind(0)
        qkv_63 = None
        q_31 = unbind_31[0]
        k_31 = unbind_31[1]
        v_31 = unbind_31[2]
        unbind_31 = None
        x_408 = torch._C._nn.scaled_dot_product_attention(
            q_31, k_31, v_31, attn_mask=None, dropout_p=0.0
        )
        q_31 = k_31 = v_31 = None
        transpose_32 = x_408.transpose(1, 2)
        x_408 = None
        x_409 = transpose_32.reshape(1, 257, 1792)
        transpose_32 = None
        x_410 = torch._C._nn.linear(
            x_409,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_,
        )
        x_409 = l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_411 = torch.nn.functional.dropout(x_410, 0.0, False, False)
        x_410 = None
        x_412 = torch.nn.functional.layer_norm(
            x_411,
            (1792,),
            l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_411 = (
            l_self_modules_blocks_modules_31_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_31_modules_norm1_parameters_bias_ = None
        x_413 = x_407 + x_412
        x_407 = x_412 = None
        x_414 = torch._C._nn.linear(
            x_413,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_415 = torch._C._nn.gelu(x_414, approximate="none")
        x_414 = None
        x_416 = torch.nn.functional.dropout(x_415, 0.0, False, False)
        x_415 = None
        x_417 = torch._C._nn.linear(
            x_416,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_416 = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_418 = torch.nn.functional.dropout(x_417, 0.0, False, False)
        x_417 = None
        x_419 = torch.nn.functional.layer_norm(
            x_418,
            (1792,),
            l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_418 = (
            l_self_modules_blocks_modules_31_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_31_modules_norm2_parameters_bias_ = None
        x_420 = x_413 + x_419
        x_413 = x_419 = None
        qkv_bias_32 = torch.cat(
            (
                l_self_modules_blocks_modules_32_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_32_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_32_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_32_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_32_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_32_modules_attn_parameters_v_bias_ = None
        qkv_64 = torch._C._nn.linear(
            x_420,
            weight=l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_32,
        )
        l_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_32
        ) = None
        reshape_64 = qkv_64.reshape(1, 257, 3, 16, -1)
        qkv_64 = None
        qkv_65 = reshape_64.permute(2, 0, 3, 1, 4)
        reshape_64 = None
        unbind_32 = qkv_65.unbind(0)
        qkv_65 = None
        q_32 = unbind_32[0]
        k_32 = unbind_32[1]
        v_32 = unbind_32[2]
        unbind_32 = None
        x_421 = torch._C._nn.scaled_dot_product_attention(
            q_32, k_32, v_32, attn_mask=None, dropout_p=0.0
        )
        q_32 = k_32 = v_32 = None
        transpose_33 = x_421.transpose(1, 2)
        x_421 = None
        x_422 = transpose_33.reshape(1, 257, 1792)
        transpose_33 = None
        x_423 = torch._C._nn.linear(
            x_422,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_,
        )
        x_422 = l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_424 = torch.nn.functional.dropout(x_423, 0.0, False, False)
        x_423 = None
        x_425 = torch.nn.functional.layer_norm(
            x_424,
            (1792,),
            l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_424 = (
            l_self_modules_blocks_modules_32_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_32_modules_norm1_parameters_bias_ = None
        x_426 = x_420 + x_425
        x_420 = x_425 = None
        x_427 = torch._C._nn.linear(
            x_426,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_428 = torch._C._nn.gelu(x_427, approximate="none")
        x_427 = None
        x_429 = torch.nn.functional.dropout(x_428, 0.0, False, False)
        x_428 = None
        x_430 = torch._C._nn.linear(
            x_429,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_429 = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_431 = torch.nn.functional.dropout(x_430, 0.0, False, False)
        x_430 = None
        x_432 = torch.nn.functional.layer_norm(
            x_431,
            (1792,),
            l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_431 = (
            l_self_modules_blocks_modules_32_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_32_modules_norm2_parameters_bias_ = None
        x_433 = x_426 + x_432
        x_426 = x_432 = None
        qkv_bias_33 = torch.cat(
            (
                l_self_modules_blocks_modules_33_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_33_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_33_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_33_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_33_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_33_modules_attn_parameters_v_bias_ = None
        qkv_66 = torch._C._nn.linear(
            x_433,
            weight=l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_33,
        )
        l_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_33
        ) = None
        reshape_66 = qkv_66.reshape(1, 257, 3, 16, -1)
        qkv_66 = None
        qkv_67 = reshape_66.permute(2, 0, 3, 1, 4)
        reshape_66 = None
        unbind_33 = qkv_67.unbind(0)
        qkv_67 = None
        q_33 = unbind_33[0]
        k_33 = unbind_33[1]
        v_33 = unbind_33[2]
        unbind_33 = None
        x_434 = torch._C._nn.scaled_dot_product_attention(
            q_33, k_33, v_33, attn_mask=None, dropout_p=0.0
        )
        q_33 = k_33 = v_33 = None
        transpose_34 = x_434.transpose(1, 2)
        x_434 = None
        x_435 = transpose_34.reshape(1, 257, 1792)
        transpose_34 = None
        x_436 = torch._C._nn.linear(
            x_435,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_,
        )
        x_435 = l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_437 = torch.nn.functional.dropout(x_436, 0.0, False, False)
        x_436 = None
        x_438 = torch.nn.functional.layer_norm(
            x_437,
            (1792,),
            l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_437 = (
            l_self_modules_blocks_modules_33_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_33_modules_norm1_parameters_bias_ = None
        x_439 = x_433 + x_438
        x_433 = x_438 = None
        x_440 = torch._C._nn.linear(
            x_439,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_441 = torch._C._nn.gelu(x_440, approximate="none")
        x_440 = None
        x_442 = torch.nn.functional.dropout(x_441, 0.0, False, False)
        x_441 = None
        x_443 = torch._C._nn.linear(
            x_442,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_442 = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_444 = torch.nn.functional.dropout(x_443, 0.0, False, False)
        x_443 = None
        x_445 = torch.nn.functional.layer_norm(
            x_444,
            (1792,),
            l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_444 = (
            l_self_modules_blocks_modules_33_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_33_modules_norm2_parameters_bias_ = None
        x_446 = x_439 + x_445
        x_439 = x_445 = None
        qkv_bias_34 = torch.cat(
            (
                l_self_modules_blocks_modules_34_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_34_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_34_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_34_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_34_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_34_modules_attn_parameters_v_bias_ = None
        qkv_68 = torch._C._nn.linear(
            x_446,
            weight=l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_34,
        )
        l_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_34
        ) = None
        reshape_68 = qkv_68.reshape(1, 257, 3, 16, -1)
        qkv_68 = None
        qkv_69 = reshape_68.permute(2, 0, 3, 1, 4)
        reshape_68 = None
        unbind_34 = qkv_69.unbind(0)
        qkv_69 = None
        q_34 = unbind_34[0]
        k_34 = unbind_34[1]
        v_34 = unbind_34[2]
        unbind_34 = None
        x_447 = torch._C._nn.scaled_dot_product_attention(
            q_34, k_34, v_34, attn_mask=None, dropout_p=0.0
        )
        q_34 = k_34 = v_34 = None
        transpose_35 = x_447.transpose(1, 2)
        x_447 = None
        x_448 = transpose_35.reshape(1, 257, 1792)
        transpose_35 = None
        x_449 = torch._C._nn.linear(
            x_448,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_,
        )
        x_448 = l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_450 = torch.nn.functional.dropout(x_449, 0.0, False, False)
        x_449 = None
        x_451 = torch.nn.functional.layer_norm(
            x_450,
            (1792,),
            l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_450 = (
            l_self_modules_blocks_modules_34_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_34_modules_norm1_parameters_bias_ = None
        x_452 = x_446 + x_451
        x_446 = x_451 = None
        x_453 = torch._C._nn.linear(
            x_452,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_454 = torch._C._nn.gelu(x_453, approximate="none")
        x_453 = None
        x_455 = torch.nn.functional.dropout(x_454, 0.0, False, False)
        x_454 = None
        x_456 = torch._C._nn.linear(
            x_455,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_455 = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_457 = torch.nn.functional.dropout(x_456, 0.0, False, False)
        x_456 = None
        x_458 = torch.nn.functional.layer_norm(
            x_457,
            (1792,),
            l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_457 = (
            l_self_modules_blocks_modules_34_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_34_modules_norm2_parameters_bias_ = None
        x_459 = x_452 + x_458
        x_452 = x_458 = None
        qkv_bias_35 = torch.cat(
            (
                l_self_modules_blocks_modules_35_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_35_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_35_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_35_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_35_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_35_modules_attn_parameters_v_bias_ = None
        qkv_70 = torch._C._nn.linear(
            x_459,
            weight=l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_35,
        )
        l_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_35
        ) = None
        reshape_70 = qkv_70.reshape(1, 257, 3, 16, -1)
        qkv_70 = None
        qkv_71 = reshape_70.permute(2, 0, 3, 1, 4)
        reshape_70 = None
        unbind_35 = qkv_71.unbind(0)
        qkv_71 = None
        q_35 = unbind_35[0]
        k_35 = unbind_35[1]
        v_35 = unbind_35[2]
        unbind_35 = None
        x_460 = torch._C._nn.scaled_dot_product_attention(
            q_35, k_35, v_35, attn_mask=None, dropout_p=0.0
        )
        q_35 = k_35 = v_35 = None
        transpose_36 = x_460.transpose(1, 2)
        x_460 = None
        x_461 = transpose_36.reshape(1, 257, 1792)
        transpose_36 = None
        x_462 = torch._C._nn.linear(
            x_461,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_,
        )
        x_461 = l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_463 = torch.nn.functional.dropout(x_462, 0.0, False, False)
        x_462 = None
        x_464 = torch.nn.functional.layer_norm(
            x_463,
            (1792,),
            l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_463 = (
            l_self_modules_blocks_modules_35_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_35_modules_norm1_parameters_bias_ = None
        x_465 = x_459 + x_464
        x_459 = x_464 = None
        x_466 = torch._C._nn.linear(
            x_465,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_467 = torch._C._nn.gelu(x_466, approximate="none")
        x_466 = None
        x_468 = torch.nn.functional.dropout(x_467, 0.0, False, False)
        x_467 = None
        x_469 = torch._C._nn.linear(
            x_468,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_468 = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_470 = torch.nn.functional.dropout(x_469, 0.0, False, False)
        x_469 = None
        x_471 = torch.nn.functional.layer_norm(
            x_470,
            (1792,),
            l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_470 = (
            l_self_modules_blocks_modules_35_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_35_modules_norm2_parameters_bias_ = None
        x_472 = x_465 + x_471
        x_465 = x_471 = None
        qkv_bias_36 = torch.cat(
            (
                l_self_modules_blocks_modules_36_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_36_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_36_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_36_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_36_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_36_modules_attn_parameters_v_bias_ = None
        qkv_72 = torch._C._nn.linear(
            x_472,
            weight=l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_36,
        )
        l_self_modules_blocks_modules_36_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_36
        ) = None
        reshape_72 = qkv_72.reshape(1, 257, 3, 16, -1)
        qkv_72 = None
        qkv_73 = reshape_72.permute(2, 0, 3, 1, 4)
        reshape_72 = None
        unbind_36 = qkv_73.unbind(0)
        qkv_73 = None
        q_36 = unbind_36[0]
        k_36 = unbind_36[1]
        v_36 = unbind_36[2]
        unbind_36 = None
        x_473 = torch._C._nn.scaled_dot_product_attention(
            q_36, k_36, v_36, attn_mask=None, dropout_p=0.0
        )
        q_36 = k_36 = v_36 = None
        transpose_37 = x_473.transpose(1, 2)
        x_473 = None
        x_474 = transpose_37.reshape(1, 257, 1792)
        transpose_37 = None
        x_475 = torch._C._nn.linear(
            x_474,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_,
        )
        x_474 = l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_476 = torch.nn.functional.dropout(x_475, 0.0, False, False)
        x_475 = None
        x_477 = torch.nn.functional.layer_norm(
            x_476,
            (1792,),
            l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_476 = (
            l_self_modules_blocks_modules_36_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_36_modules_norm1_parameters_bias_ = None
        x_478 = x_472 + x_477
        x_472 = x_477 = None
        x_479 = torch._C._nn.linear(
            x_478,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_480 = torch._C._nn.gelu(x_479, approximate="none")
        x_479 = None
        x_481 = torch.nn.functional.dropout(x_480, 0.0, False, False)
        x_480 = None
        x_482 = torch._C._nn.linear(
            x_481,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_481 = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_36_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_483 = torch.nn.functional.dropout(x_482, 0.0, False, False)
        x_482 = None
        x_484 = torch.nn.functional.layer_norm(
            x_483,
            (1792,),
            l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_483 = (
            l_self_modules_blocks_modules_36_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_36_modules_norm2_parameters_bias_ = None
        x_485 = x_478 + x_484
        x_478 = x_484 = None
        qkv_bias_37 = torch.cat(
            (
                l_self_modules_blocks_modules_37_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_37_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_37_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_37_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_37_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_37_modules_attn_parameters_v_bias_ = None
        qkv_74 = torch._C._nn.linear(
            x_485,
            weight=l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_37,
        )
        l_self_modules_blocks_modules_37_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_37
        ) = None
        reshape_74 = qkv_74.reshape(1, 257, 3, 16, -1)
        qkv_74 = None
        qkv_75 = reshape_74.permute(2, 0, 3, 1, 4)
        reshape_74 = None
        unbind_37 = qkv_75.unbind(0)
        qkv_75 = None
        q_37 = unbind_37[0]
        k_37 = unbind_37[1]
        v_37 = unbind_37[2]
        unbind_37 = None
        x_486 = torch._C._nn.scaled_dot_product_attention(
            q_37, k_37, v_37, attn_mask=None, dropout_p=0.0
        )
        q_37 = k_37 = v_37 = None
        transpose_38 = x_486.transpose(1, 2)
        x_486 = None
        x_487 = transpose_38.reshape(1, 257, 1792)
        transpose_38 = None
        x_488 = torch._C._nn.linear(
            x_487,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_,
        )
        x_487 = l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_489 = torch.nn.functional.dropout(x_488, 0.0, False, False)
        x_488 = None
        x_490 = torch.nn.functional.layer_norm(
            x_489,
            (1792,),
            l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_489 = (
            l_self_modules_blocks_modules_37_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_37_modules_norm1_parameters_bias_ = None
        x_491 = x_485 + x_490
        x_485 = x_490 = None
        x_492 = torch._C._nn.linear(
            x_491,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_493 = torch._C._nn.gelu(x_492, approximate="none")
        x_492 = None
        x_494 = torch.nn.functional.dropout(x_493, 0.0, False, False)
        x_493 = None
        x_495 = torch._C._nn.linear(
            x_494,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_494 = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_37_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_496 = torch.nn.functional.dropout(x_495, 0.0, False, False)
        x_495 = None
        x_497 = torch.nn.functional.layer_norm(
            x_496,
            (1792,),
            l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_496 = (
            l_self_modules_blocks_modules_37_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_37_modules_norm2_parameters_bias_ = None
        x_498 = x_491 + x_497
        x_491 = x_497 = None
        qkv_bias_38 = torch.cat(
            (
                l_self_modules_blocks_modules_38_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_38_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_38_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_38_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_38_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_38_modules_attn_parameters_v_bias_ = None
        qkv_76 = torch._C._nn.linear(
            x_498,
            weight=l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_38,
        )
        l_self_modules_blocks_modules_38_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_38
        ) = None
        reshape_76 = qkv_76.reshape(1, 257, 3, 16, -1)
        qkv_76 = None
        qkv_77 = reshape_76.permute(2, 0, 3, 1, 4)
        reshape_76 = None
        unbind_38 = qkv_77.unbind(0)
        qkv_77 = None
        q_38 = unbind_38[0]
        k_38 = unbind_38[1]
        v_38 = unbind_38[2]
        unbind_38 = None
        x_499 = torch._C._nn.scaled_dot_product_attention(
            q_38, k_38, v_38, attn_mask=None, dropout_p=0.0
        )
        q_38 = k_38 = v_38 = None
        transpose_39 = x_499.transpose(1, 2)
        x_499 = None
        x_500 = transpose_39.reshape(1, 257, 1792)
        transpose_39 = None
        x_501 = torch._C._nn.linear(
            x_500,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_,
        )
        x_500 = l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_502 = torch.nn.functional.dropout(x_501, 0.0, False, False)
        x_501 = None
        x_503 = torch.nn.functional.layer_norm(
            x_502,
            (1792,),
            l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_502 = (
            l_self_modules_blocks_modules_38_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_38_modules_norm1_parameters_bias_ = None
        x_504 = x_498 + x_503
        x_498 = x_503 = None
        x_505 = torch._C._nn.linear(
            x_504,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_506 = torch._C._nn.gelu(x_505, approximate="none")
        x_505 = None
        x_507 = torch.nn.functional.dropout(x_506, 0.0, False, False)
        x_506 = None
        x_508 = torch._C._nn.linear(
            x_507,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_507 = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_38_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_509 = torch.nn.functional.dropout(x_508, 0.0, False, False)
        x_508 = None
        x_510 = torch.nn.functional.layer_norm(
            x_509,
            (1792,),
            l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_509 = (
            l_self_modules_blocks_modules_38_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_38_modules_norm2_parameters_bias_ = None
        x_511 = x_504 + x_510
        x_504 = x_510 = None
        qkv_bias_39 = torch.cat(
            (
                l_self_modules_blocks_modules_39_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_39_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_39_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_39_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_39_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_39_modules_attn_parameters_v_bias_ = None
        qkv_78 = torch._C._nn.linear(
            x_511,
            weight=l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_39,
        )
        l_self_modules_blocks_modules_39_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_39
        ) = None
        reshape_78 = qkv_78.reshape(1, 257, 3, 16, -1)
        qkv_78 = None
        qkv_79 = reshape_78.permute(2, 0, 3, 1, 4)
        reshape_78 = None
        unbind_39 = qkv_79.unbind(0)
        qkv_79 = None
        q_39 = unbind_39[0]
        k_39 = unbind_39[1]
        v_39 = unbind_39[2]
        unbind_39 = None
        x_512 = torch._C._nn.scaled_dot_product_attention(
            q_39, k_39, v_39, attn_mask=None, dropout_p=0.0
        )
        q_39 = k_39 = v_39 = None
        transpose_40 = x_512.transpose(1, 2)
        x_512 = None
        x_513 = transpose_40.reshape(1, 257, 1792)
        transpose_40 = None
        x_514 = torch._C._nn.linear(
            x_513,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_,
        )
        x_513 = l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_515 = torch.nn.functional.dropout(x_514, 0.0, False, False)
        x_514 = None
        x_516 = torch.nn.functional.layer_norm(
            x_515,
            (1792,),
            l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_515 = (
            l_self_modules_blocks_modules_39_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_39_modules_norm1_parameters_bias_ = None
        x_517 = x_511 + x_516
        x_511 = x_516 = None
        x_518 = torch._C._nn.linear(
            x_517,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_519 = torch._C._nn.gelu(x_518, approximate="none")
        x_518 = None
        x_520 = torch.nn.functional.dropout(x_519, 0.0, False, False)
        x_519 = None
        x_521 = torch._C._nn.linear(
            x_520,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_520 = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_39_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_522 = torch.nn.functional.dropout(x_521, 0.0, False, False)
        x_521 = None
        x_523 = torch.nn.functional.layer_norm(
            x_522,
            (1792,),
            l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_522 = (
            l_self_modules_blocks_modules_39_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_39_modules_norm2_parameters_bias_ = None
        x_524 = x_517 + x_523
        x_517 = x_523 = None
        qkv_bias_40 = torch.cat(
            (
                l_self_modules_blocks_modules_40_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_40_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_40_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_40_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_40_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_40_modules_attn_parameters_v_bias_ = None
        qkv_80 = torch._C._nn.linear(
            x_524,
            weight=l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_40,
        )
        l_self_modules_blocks_modules_40_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_40
        ) = None
        reshape_80 = qkv_80.reshape(1, 257, 3, 16, -1)
        qkv_80 = None
        qkv_81 = reshape_80.permute(2, 0, 3, 1, 4)
        reshape_80 = None
        unbind_40 = qkv_81.unbind(0)
        qkv_81 = None
        q_40 = unbind_40[0]
        k_40 = unbind_40[1]
        v_40 = unbind_40[2]
        unbind_40 = None
        x_525 = torch._C._nn.scaled_dot_product_attention(
            q_40, k_40, v_40, attn_mask=None, dropout_p=0.0
        )
        q_40 = k_40 = v_40 = None
        transpose_41 = x_525.transpose(1, 2)
        x_525 = None
        x_526 = transpose_41.reshape(1, 257, 1792)
        transpose_41 = None
        x_527 = torch._C._nn.linear(
            x_526,
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_,
        )
        x_526 = l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_40_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_528 = torch.nn.functional.dropout(x_527, 0.0, False, False)
        x_527 = None
        x_529 = torch.nn.functional.layer_norm(
            x_528,
            (1792,),
            l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_528 = (
            l_self_modules_blocks_modules_40_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_40_modules_norm1_parameters_bias_ = None
        x_530 = x_524 + x_529
        x_524 = x_529 = None
        x_531 = torch._C._nn.linear(
            x_530,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_532 = torch._C._nn.gelu(x_531, approximate="none")
        x_531 = None
        x_533 = torch.nn.functional.dropout(x_532, 0.0, False, False)
        x_532 = None
        x_534 = torch._C._nn.linear(
            x_533,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_533 = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_40_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_535 = torch.nn.functional.dropout(x_534, 0.0, False, False)
        x_534 = None
        x_536 = torch.nn.functional.layer_norm(
            x_535,
            (1792,),
            l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_535 = (
            l_self_modules_blocks_modules_40_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_40_modules_norm2_parameters_bias_ = None
        x_537 = x_530 + x_536
        x_530 = x_536 = None
        qkv_bias_41 = torch.cat(
            (
                l_self_modules_blocks_modules_41_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_41_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_41_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_41_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_41_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_41_modules_attn_parameters_v_bias_ = None
        qkv_82 = torch._C._nn.linear(
            x_537,
            weight=l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_41,
        )
        l_self_modules_blocks_modules_41_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_41
        ) = None
        reshape_82 = qkv_82.reshape(1, 257, 3, 16, -1)
        qkv_82 = None
        qkv_83 = reshape_82.permute(2, 0, 3, 1, 4)
        reshape_82 = None
        unbind_41 = qkv_83.unbind(0)
        qkv_83 = None
        q_41 = unbind_41[0]
        k_41 = unbind_41[1]
        v_41 = unbind_41[2]
        unbind_41 = None
        x_538 = torch._C._nn.scaled_dot_product_attention(
            q_41, k_41, v_41, attn_mask=None, dropout_p=0.0
        )
        q_41 = k_41 = v_41 = None
        transpose_42 = x_538.transpose(1, 2)
        x_538 = None
        x_539 = transpose_42.reshape(1, 257, 1792)
        transpose_42 = None
        x_540 = torch._C._nn.linear(
            x_539,
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_,
        )
        x_539 = l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_41_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_541 = torch.nn.functional.dropout(x_540, 0.0, False, False)
        x_540 = None
        x_542 = torch.nn.functional.layer_norm(
            x_541,
            (1792,),
            l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_541 = (
            l_self_modules_blocks_modules_41_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_41_modules_norm1_parameters_bias_ = None
        x_543 = x_537 + x_542
        x_537 = x_542 = None
        x_544 = torch._C._nn.linear(
            x_543,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_545 = torch._C._nn.gelu(x_544, approximate="none")
        x_544 = None
        x_546 = torch.nn.functional.dropout(x_545, 0.0, False, False)
        x_545 = None
        x_547 = torch._C._nn.linear(
            x_546,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_546 = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_41_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_548 = torch.nn.functional.dropout(x_547, 0.0, False, False)
        x_547 = None
        x_549 = torch.nn.functional.layer_norm(
            x_548,
            (1792,),
            l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_548 = (
            l_self_modules_blocks_modules_41_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_41_modules_norm2_parameters_bias_ = None
        x_550 = x_543 + x_549
        x_543 = x_549 = None
        qkv_bias_42 = torch.cat(
            (
                l_self_modules_blocks_modules_42_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_42_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_42_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_42_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_42_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_42_modules_attn_parameters_v_bias_ = None
        qkv_84 = torch._C._nn.linear(
            x_550,
            weight=l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_42,
        )
        l_self_modules_blocks_modules_42_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_42
        ) = None
        reshape_84 = qkv_84.reshape(1, 257, 3, 16, -1)
        qkv_84 = None
        qkv_85 = reshape_84.permute(2, 0, 3, 1, 4)
        reshape_84 = None
        unbind_42 = qkv_85.unbind(0)
        qkv_85 = None
        q_42 = unbind_42[0]
        k_42 = unbind_42[1]
        v_42 = unbind_42[2]
        unbind_42 = None
        x_551 = torch._C._nn.scaled_dot_product_attention(
            q_42, k_42, v_42, attn_mask=None, dropout_p=0.0
        )
        q_42 = k_42 = v_42 = None
        transpose_43 = x_551.transpose(1, 2)
        x_551 = None
        x_552 = transpose_43.reshape(1, 257, 1792)
        transpose_43 = None
        x_553 = torch._C._nn.linear(
            x_552,
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_,
        )
        x_552 = l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_42_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_554 = torch.nn.functional.dropout(x_553, 0.0, False, False)
        x_553 = None
        x_555 = torch.nn.functional.layer_norm(
            x_554,
            (1792,),
            l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_554 = (
            l_self_modules_blocks_modules_42_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_42_modules_norm1_parameters_bias_ = None
        x_556 = x_550 + x_555
        x_550 = x_555 = None
        x_557 = torch._C._nn.linear(
            x_556,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_558 = torch._C._nn.gelu(x_557, approximate="none")
        x_557 = None
        x_559 = torch.nn.functional.dropout(x_558, 0.0, False, False)
        x_558 = None
        x_560 = torch._C._nn.linear(
            x_559,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_559 = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_42_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_561 = torch.nn.functional.dropout(x_560, 0.0, False, False)
        x_560 = None
        x_562 = torch.nn.functional.layer_norm(
            x_561,
            (1792,),
            l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_561 = (
            l_self_modules_blocks_modules_42_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_42_modules_norm2_parameters_bias_ = None
        x_563 = x_556 + x_562
        x_556 = x_562 = None
        qkv_bias_43 = torch.cat(
            (
                l_self_modules_blocks_modules_43_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_43_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_43_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_43_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_43_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_43_modules_attn_parameters_v_bias_ = None
        qkv_86 = torch._C._nn.linear(
            x_563,
            weight=l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_43,
        )
        l_self_modules_blocks_modules_43_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_43
        ) = None
        reshape_86 = qkv_86.reshape(1, 257, 3, 16, -1)
        qkv_86 = None
        qkv_87 = reshape_86.permute(2, 0, 3, 1, 4)
        reshape_86 = None
        unbind_43 = qkv_87.unbind(0)
        qkv_87 = None
        q_43 = unbind_43[0]
        k_43 = unbind_43[1]
        v_43 = unbind_43[2]
        unbind_43 = None
        x_564 = torch._C._nn.scaled_dot_product_attention(
            q_43, k_43, v_43, attn_mask=None, dropout_p=0.0
        )
        q_43 = k_43 = v_43 = None
        transpose_44 = x_564.transpose(1, 2)
        x_564 = None
        x_565 = transpose_44.reshape(1, 257, 1792)
        transpose_44 = None
        x_566 = torch._C._nn.linear(
            x_565,
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_,
        )
        x_565 = l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_43_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_567 = torch.nn.functional.dropout(x_566, 0.0, False, False)
        x_566 = None
        x_568 = torch.nn.functional.layer_norm(
            x_567,
            (1792,),
            l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_567 = (
            l_self_modules_blocks_modules_43_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_43_modules_norm1_parameters_bias_ = None
        x_569 = x_563 + x_568
        x_563 = x_568 = None
        x_570 = torch._C._nn.linear(
            x_569,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_571 = torch._C._nn.gelu(x_570, approximate="none")
        x_570 = None
        x_572 = torch.nn.functional.dropout(x_571, 0.0, False, False)
        x_571 = None
        x_573 = torch._C._nn.linear(
            x_572,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_572 = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_43_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_574 = torch.nn.functional.dropout(x_573, 0.0, False, False)
        x_573 = None
        x_575 = torch.nn.functional.layer_norm(
            x_574,
            (1792,),
            l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_574 = (
            l_self_modules_blocks_modules_43_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_43_modules_norm2_parameters_bias_ = None
        x_576 = x_569 + x_575
        x_569 = x_575 = None
        qkv_bias_44 = torch.cat(
            (
                l_self_modules_blocks_modules_44_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_44_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_44_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_44_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_44_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_44_modules_attn_parameters_v_bias_ = None
        qkv_88 = torch._C._nn.linear(
            x_576,
            weight=l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_44,
        )
        l_self_modules_blocks_modules_44_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_44
        ) = None
        reshape_88 = qkv_88.reshape(1, 257, 3, 16, -1)
        qkv_88 = None
        qkv_89 = reshape_88.permute(2, 0, 3, 1, 4)
        reshape_88 = None
        unbind_44 = qkv_89.unbind(0)
        qkv_89 = None
        q_44 = unbind_44[0]
        k_44 = unbind_44[1]
        v_44 = unbind_44[2]
        unbind_44 = None
        x_577 = torch._C._nn.scaled_dot_product_attention(
            q_44, k_44, v_44, attn_mask=None, dropout_p=0.0
        )
        q_44 = k_44 = v_44 = None
        transpose_45 = x_577.transpose(1, 2)
        x_577 = None
        x_578 = transpose_45.reshape(1, 257, 1792)
        transpose_45 = None
        x_579 = torch._C._nn.linear(
            x_578,
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_,
        )
        x_578 = l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_44_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_580 = torch.nn.functional.dropout(x_579, 0.0, False, False)
        x_579 = None
        x_581 = torch.nn.functional.layer_norm(
            x_580,
            (1792,),
            l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_580 = (
            l_self_modules_blocks_modules_44_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_44_modules_norm1_parameters_bias_ = None
        x_582 = x_576 + x_581
        x_576 = x_581 = None
        x_583 = torch._C._nn.linear(
            x_582,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_584 = torch._C._nn.gelu(x_583, approximate="none")
        x_583 = None
        x_585 = torch.nn.functional.dropout(x_584, 0.0, False, False)
        x_584 = None
        x_586 = torch._C._nn.linear(
            x_585,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_585 = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_44_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_587 = torch.nn.functional.dropout(x_586, 0.0, False, False)
        x_586 = None
        x_588 = torch.nn.functional.layer_norm(
            x_587,
            (1792,),
            l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_587 = (
            l_self_modules_blocks_modules_44_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_44_modules_norm2_parameters_bias_ = None
        x_589 = x_582 + x_588
        x_582 = x_588 = None
        qkv_bias_45 = torch.cat(
            (
                l_self_modules_blocks_modules_45_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_45_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_45_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_45_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_45_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_45_modules_attn_parameters_v_bias_ = None
        qkv_90 = torch._C._nn.linear(
            x_589,
            weight=l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_45,
        )
        l_self_modules_blocks_modules_45_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_45
        ) = None
        reshape_90 = qkv_90.reshape(1, 257, 3, 16, -1)
        qkv_90 = None
        qkv_91 = reshape_90.permute(2, 0, 3, 1, 4)
        reshape_90 = None
        unbind_45 = qkv_91.unbind(0)
        qkv_91 = None
        q_45 = unbind_45[0]
        k_45 = unbind_45[1]
        v_45 = unbind_45[2]
        unbind_45 = None
        x_590 = torch._C._nn.scaled_dot_product_attention(
            q_45, k_45, v_45, attn_mask=None, dropout_p=0.0
        )
        q_45 = k_45 = v_45 = None
        transpose_46 = x_590.transpose(1, 2)
        x_590 = None
        x_591 = transpose_46.reshape(1, 257, 1792)
        transpose_46 = None
        x_592 = torch._C._nn.linear(
            x_591,
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_,
        )
        x_591 = l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_45_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_593 = torch.nn.functional.dropout(x_592, 0.0, False, False)
        x_592 = None
        x_594 = torch.nn.functional.layer_norm(
            x_593,
            (1792,),
            l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_593 = (
            l_self_modules_blocks_modules_45_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_45_modules_norm1_parameters_bias_ = None
        x_595 = x_589 + x_594
        x_589 = x_594 = None
        x_596 = torch._C._nn.linear(
            x_595,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_597 = torch._C._nn.gelu(x_596, approximate="none")
        x_596 = None
        x_598 = torch.nn.functional.dropout(x_597, 0.0, False, False)
        x_597 = None
        x_599 = torch._C._nn.linear(
            x_598,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_598 = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_45_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_600 = torch.nn.functional.dropout(x_599, 0.0, False, False)
        x_599 = None
        x_601 = torch.nn.functional.layer_norm(
            x_600,
            (1792,),
            l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_600 = (
            l_self_modules_blocks_modules_45_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_45_modules_norm2_parameters_bias_ = None
        x_602 = x_595 + x_601
        x_595 = x_601 = None
        qkv_bias_46 = torch.cat(
            (
                l_self_modules_blocks_modules_46_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_46_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_46_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_46_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_46_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_46_modules_attn_parameters_v_bias_ = None
        qkv_92 = torch._C._nn.linear(
            x_602,
            weight=l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_46,
        )
        l_self_modules_blocks_modules_46_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_46
        ) = None
        reshape_92 = qkv_92.reshape(1, 257, 3, 16, -1)
        qkv_92 = None
        qkv_93 = reshape_92.permute(2, 0, 3, 1, 4)
        reshape_92 = None
        unbind_46 = qkv_93.unbind(0)
        qkv_93 = None
        q_46 = unbind_46[0]
        k_46 = unbind_46[1]
        v_46 = unbind_46[2]
        unbind_46 = None
        x_603 = torch._C._nn.scaled_dot_product_attention(
            q_46, k_46, v_46, attn_mask=None, dropout_p=0.0
        )
        q_46 = k_46 = v_46 = None
        transpose_47 = x_603.transpose(1, 2)
        x_603 = None
        x_604 = transpose_47.reshape(1, 257, 1792)
        transpose_47 = None
        x_605 = torch._C._nn.linear(
            x_604,
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_,
        )
        x_604 = l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_46_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_606 = torch.nn.functional.dropout(x_605, 0.0, False, False)
        x_605 = None
        x_607 = torch.nn.functional.layer_norm(
            x_606,
            (1792,),
            l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_606 = (
            l_self_modules_blocks_modules_46_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_46_modules_norm1_parameters_bias_ = None
        x_608 = x_602 + x_607
        x_602 = x_607 = None
        x_609 = torch._C._nn.linear(
            x_608,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_610 = torch._C._nn.gelu(x_609, approximate="none")
        x_609 = None
        x_611 = torch.nn.functional.dropout(x_610, 0.0, False, False)
        x_610 = None
        x_612 = torch._C._nn.linear(
            x_611,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_611 = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_46_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_613 = torch.nn.functional.dropout(x_612, 0.0, False, False)
        x_612 = None
        x_614 = torch.nn.functional.layer_norm(
            x_613,
            (1792,),
            l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_613 = (
            l_self_modules_blocks_modules_46_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_46_modules_norm2_parameters_bias_ = None
        x_615 = x_608 + x_614
        x_608 = x_614 = None
        qkv_bias_47 = torch.cat(
            (
                l_self_modules_blocks_modules_47_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_47_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_47_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_47_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_47_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_47_modules_attn_parameters_v_bias_ = None
        qkv_94 = torch._C._nn.linear(
            x_615,
            weight=l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_47,
        )
        l_self_modules_blocks_modules_47_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_47
        ) = None
        reshape_94 = qkv_94.reshape(1, 257, 3, 16, -1)
        qkv_94 = None
        qkv_95 = reshape_94.permute(2, 0, 3, 1, 4)
        reshape_94 = None
        unbind_47 = qkv_95.unbind(0)
        qkv_95 = None
        q_47 = unbind_47[0]
        k_47 = unbind_47[1]
        v_47 = unbind_47[2]
        unbind_47 = None
        x_616 = torch._C._nn.scaled_dot_product_attention(
            q_47, k_47, v_47, attn_mask=None, dropout_p=0.0
        )
        q_47 = k_47 = v_47 = None
        transpose_48 = x_616.transpose(1, 2)
        x_616 = None
        x_617 = transpose_48.reshape(1, 257, 1792)
        transpose_48 = None
        x_618 = torch._C._nn.linear(
            x_617,
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_,
        )
        x_617 = l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_47_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_619 = torch.nn.functional.dropout(x_618, 0.0, False, False)
        x_618 = None
        x_620 = torch.nn.functional.layer_norm(
            x_619,
            (1792,),
            l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_619 = (
            l_self_modules_blocks_modules_47_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_47_modules_norm1_parameters_bias_ = None
        x_621 = x_615 + x_620
        x_615 = x_620 = None
        x_622 = torch._C._nn.linear(
            x_621,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_623 = torch._C._nn.gelu(x_622, approximate="none")
        x_622 = None
        x_624 = torch.nn.functional.dropout(x_623, 0.0, False, False)
        x_623 = None
        x_625 = torch._C._nn.linear(
            x_624,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_624 = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_47_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_626 = torch.nn.functional.dropout(x_625, 0.0, False, False)
        x_625 = None
        x_627 = torch.nn.functional.layer_norm(
            x_626,
            (1792,),
            l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_626 = (
            l_self_modules_blocks_modules_47_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_47_modules_norm2_parameters_bias_ = None
        x_628 = x_621 + x_627
        x_621 = x_627 = None
        qkv_bias_48 = torch.cat(
            (
                l_self_modules_blocks_modules_48_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_48_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_48_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_48_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_48_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_48_modules_attn_parameters_v_bias_ = None
        qkv_96 = torch._C._nn.linear(
            x_628,
            weight=l_self_modules_blocks_modules_48_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_48,
        )
        l_self_modules_blocks_modules_48_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_48
        ) = None
        reshape_96 = qkv_96.reshape(1, 257, 3, 16, -1)
        qkv_96 = None
        qkv_97 = reshape_96.permute(2, 0, 3, 1, 4)
        reshape_96 = None
        unbind_48 = qkv_97.unbind(0)
        qkv_97 = None
        q_48 = unbind_48[0]
        k_48 = unbind_48[1]
        v_48 = unbind_48[2]
        unbind_48 = None
        x_629 = torch._C._nn.scaled_dot_product_attention(
            q_48, k_48, v_48, attn_mask=None, dropout_p=0.0
        )
        q_48 = k_48 = v_48 = None
        transpose_49 = x_629.transpose(1, 2)
        x_629 = None
        x_630 = transpose_49.reshape(1, 257, 1792)
        transpose_49 = None
        x_631 = torch._C._nn.linear(
            x_630,
            l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_bias_,
        )
        x_630 = l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_48_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_632 = torch.nn.functional.dropout(x_631, 0.0, False, False)
        x_631 = None
        x_633 = torch.nn.functional.layer_norm(
            x_632,
            (1792,),
            l_self_modules_blocks_modules_48_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_48_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_632 = (
            l_self_modules_blocks_modules_48_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_48_modules_norm1_parameters_bias_ = None
        x_634 = x_628 + x_633
        x_628 = x_633 = None
        x_635 = torch._C._nn.linear(
            x_634,
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_636 = torch._C._nn.gelu(x_635, approximate="none")
        x_635 = None
        x_637 = torch.nn.functional.dropout(x_636, 0.0, False, False)
        x_636 = None
        x_638 = torch._C._nn.linear(
            x_637,
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_637 = (
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_48_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_639 = torch.nn.functional.dropout(x_638, 0.0, False, False)
        x_638 = None
        x_640 = torch.nn.functional.layer_norm(
            x_639,
            (1792,),
            l_self_modules_blocks_modules_48_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_48_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_639 = (
            l_self_modules_blocks_modules_48_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_48_modules_norm2_parameters_bias_ = None
        x_641 = x_634 + x_640
        x_634 = x_640 = None
        qkv_bias_49 = torch.cat(
            (
                l_self_modules_blocks_modules_49_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_49_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_49_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_49_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_49_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_49_modules_attn_parameters_v_bias_ = None
        qkv_98 = torch._C._nn.linear(
            x_641,
            weight=l_self_modules_blocks_modules_49_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_49,
        )
        l_self_modules_blocks_modules_49_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_49
        ) = None
        reshape_98 = qkv_98.reshape(1, 257, 3, 16, -1)
        qkv_98 = None
        qkv_99 = reshape_98.permute(2, 0, 3, 1, 4)
        reshape_98 = None
        unbind_49 = qkv_99.unbind(0)
        qkv_99 = None
        q_49 = unbind_49[0]
        k_49 = unbind_49[1]
        v_49 = unbind_49[2]
        unbind_49 = None
        x_642 = torch._C._nn.scaled_dot_product_attention(
            q_49, k_49, v_49, attn_mask=None, dropout_p=0.0
        )
        q_49 = k_49 = v_49 = None
        transpose_50 = x_642.transpose(1, 2)
        x_642 = None
        x_643 = transpose_50.reshape(1, 257, 1792)
        transpose_50 = None
        x_644 = torch._C._nn.linear(
            x_643,
            l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_bias_,
        )
        x_643 = l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_49_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_645 = torch.nn.functional.dropout(x_644, 0.0, False, False)
        x_644 = None
        x_646 = torch.nn.functional.layer_norm(
            x_645,
            (1792,),
            l_self_modules_blocks_modules_49_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_49_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_645 = (
            l_self_modules_blocks_modules_49_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_49_modules_norm1_parameters_bias_ = None
        x_647 = x_641 + x_646
        x_641 = x_646 = None
        x_648 = torch._C._nn.linear(
            x_647,
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_649 = torch._C._nn.gelu(x_648, approximate="none")
        x_648 = None
        x_650 = torch.nn.functional.dropout(x_649, 0.0, False, False)
        x_649 = None
        x_651 = torch._C._nn.linear(
            x_650,
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_650 = (
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_49_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_652 = torch.nn.functional.dropout(x_651, 0.0, False, False)
        x_651 = None
        x_653 = torch.nn.functional.layer_norm(
            x_652,
            (1792,),
            l_self_modules_blocks_modules_49_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_49_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_652 = (
            l_self_modules_blocks_modules_49_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_49_modules_norm2_parameters_bias_ = None
        x_654 = x_647 + x_653
        x_647 = x_653 = None
        qkv_bias_50 = torch.cat(
            (
                l_self_modules_blocks_modules_50_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_50_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_50_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_50_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_50_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_50_modules_attn_parameters_v_bias_ = None
        qkv_100 = torch._C._nn.linear(
            x_654,
            weight=l_self_modules_blocks_modules_50_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_50,
        )
        l_self_modules_blocks_modules_50_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_50
        ) = None
        reshape_100 = qkv_100.reshape(1, 257, 3, 16, -1)
        qkv_100 = None
        qkv_101 = reshape_100.permute(2, 0, 3, 1, 4)
        reshape_100 = None
        unbind_50 = qkv_101.unbind(0)
        qkv_101 = None
        q_50 = unbind_50[0]
        k_50 = unbind_50[1]
        v_50 = unbind_50[2]
        unbind_50 = None
        x_655 = torch._C._nn.scaled_dot_product_attention(
            q_50, k_50, v_50, attn_mask=None, dropout_p=0.0
        )
        q_50 = k_50 = v_50 = None
        transpose_51 = x_655.transpose(1, 2)
        x_655 = None
        x_656 = transpose_51.reshape(1, 257, 1792)
        transpose_51 = None
        x_657 = torch._C._nn.linear(
            x_656,
            l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_bias_,
        )
        x_656 = l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_50_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_658 = torch.nn.functional.dropout(x_657, 0.0, False, False)
        x_657 = None
        x_659 = torch.nn.functional.layer_norm(
            x_658,
            (1792,),
            l_self_modules_blocks_modules_50_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_50_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_658 = (
            l_self_modules_blocks_modules_50_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_50_modules_norm1_parameters_bias_ = None
        x_660 = x_654 + x_659
        x_654 = x_659 = None
        x_661 = torch._C._nn.linear(
            x_660,
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_662 = torch._C._nn.gelu(x_661, approximate="none")
        x_661 = None
        x_663 = torch.nn.functional.dropout(x_662, 0.0, False, False)
        x_662 = None
        x_664 = torch._C._nn.linear(
            x_663,
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_663 = (
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_50_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_665 = torch.nn.functional.dropout(x_664, 0.0, False, False)
        x_664 = None
        x_666 = torch.nn.functional.layer_norm(
            x_665,
            (1792,),
            l_self_modules_blocks_modules_50_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_50_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_665 = (
            l_self_modules_blocks_modules_50_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_50_modules_norm2_parameters_bias_ = None
        x_667 = x_660 + x_666
        x_660 = x_666 = None
        qkv_bias_51 = torch.cat(
            (
                l_self_modules_blocks_modules_51_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_51_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_51_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_51_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_51_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_51_modules_attn_parameters_v_bias_ = None
        qkv_102 = torch._C._nn.linear(
            x_667,
            weight=l_self_modules_blocks_modules_51_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_51,
        )
        l_self_modules_blocks_modules_51_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_51
        ) = None
        reshape_102 = qkv_102.reshape(1, 257, 3, 16, -1)
        qkv_102 = None
        qkv_103 = reshape_102.permute(2, 0, 3, 1, 4)
        reshape_102 = None
        unbind_51 = qkv_103.unbind(0)
        qkv_103 = None
        q_51 = unbind_51[0]
        k_51 = unbind_51[1]
        v_51 = unbind_51[2]
        unbind_51 = None
        x_668 = torch._C._nn.scaled_dot_product_attention(
            q_51, k_51, v_51, attn_mask=None, dropout_p=0.0
        )
        q_51 = k_51 = v_51 = None
        transpose_52 = x_668.transpose(1, 2)
        x_668 = None
        x_669 = transpose_52.reshape(1, 257, 1792)
        transpose_52 = None
        x_670 = torch._C._nn.linear(
            x_669,
            l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_bias_,
        )
        x_669 = l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_51_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_671 = torch.nn.functional.dropout(x_670, 0.0, False, False)
        x_670 = None
        x_672 = torch.nn.functional.layer_norm(
            x_671,
            (1792,),
            l_self_modules_blocks_modules_51_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_51_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_671 = (
            l_self_modules_blocks_modules_51_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_51_modules_norm1_parameters_bias_ = None
        x_673 = x_667 + x_672
        x_667 = x_672 = None
        x_674 = torch._C._nn.linear(
            x_673,
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_675 = torch._C._nn.gelu(x_674, approximate="none")
        x_674 = None
        x_676 = torch.nn.functional.dropout(x_675, 0.0, False, False)
        x_675 = None
        x_677 = torch._C._nn.linear(
            x_676,
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_676 = (
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_51_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_678 = torch.nn.functional.dropout(x_677, 0.0, False, False)
        x_677 = None
        x_679 = torch.nn.functional.layer_norm(
            x_678,
            (1792,),
            l_self_modules_blocks_modules_51_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_51_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_678 = (
            l_self_modules_blocks_modules_51_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_51_modules_norm2_parameters_bias_ = None
        x_680 = x_673 + x_679
        x_673 = x_679 = None
        qkv_bias_52 = torch.cat(
            (
                l_self_modules_blocks_modules_52_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_52_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_52_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_52_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_52_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_52_modules_attn_parameters_v_bias_ = None
        qkv_104 = torch._C._nn.linear(
            x_680,
            weight=l_self_modules_blocks_modules_52_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_52,
        )
        l_self_modules_blocks_modules_52_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_52
        ) = None
        reshape_104 = qkv_104.reshape(1, 257, 3, 16, -1)
        qkv_104 = None
        qkv_105 = reshape_104.permute(2, 0, 3, 1, 4)
        reshape_104 = None
        unbind_52 = qkv_105.unbind(0)
        qkv_105 = None
        q_52 = unbind_52[0]
        k_52 = unbind_52[1]
        v_52 = unbind_52[2]
        unbind_52 = None
        x_681 = torch._C._nn.scaled_dot_product_attention(
            q_52, k_52, v_52, attn_mask=None, dropout_p=0.0
        )
        q_52 = k_52 = v_52 = None
        transpose_53 = x_681.transpose(1, 2)
        x_681 = None
        x_682 = transpose_53.reshape(1, 257, 1792)
        transpose_53 = None
        x_683 = torch._C._nn.linear(
            x_682,
            l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_bias_,
        )
        x_682 = l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_52_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_684 = torch.nn.functional.dropout(x_683, 0.0, False, False)
        x_683 = None
        x_685 = torch.nn.functional.layer_norm(
            x_684,
            (1792,),
            l_self_modules_blocks_modules_52_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_52_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_684 = (
            l_self_modules_blocks_modules_52_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_52_modules_norm1_parameters_bias_ = None
        x_686 = x_680 + x_685
        x_680 = x_685 = None
        x_687 = torch._C._nn.linear(
            x_686,
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_688 = torch._C._nn.gelu(x_687, approximate="none")
        x_687 = None
        x_689 = torch.nn.functional.dropout(x_688, 0.0, False, False)
        x_688 = None
        x_690 = torch._C._nn.linear(
            x_689,
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_689 = (
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_52_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_691 = torch.nn.functional.dropout(x_690, 0.0, False, False)
        x_690 = None
        x_692 = torch.nn.functional.layer_norm(
            x_691,
            (1792,),
            l_self_modules_blocks_modules_52_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_52_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_691 = (
            l_self_modules_blocks_modules_52_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_52_modules_norm2_parameters_bias_ = None
        x_693 = x_686 + x_692
        x_686 = x_692 = None
        qkv_bias_53 = torch.cat(
            (
                l_self_modules_blocks_modules_53_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_53_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_53_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_53_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_53_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_53_modules_attn_parameters_v_bias_ = None
        qkv_106 = torch._C._nn.linear(
            x_693,
            weight=l_self_modules_blocks_modules_53_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_53,
        )
        l_self_modules_blocks_modules_53_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_53
        ) = None
        reshape_106 = qkv_106.reshape(1, 257, 3, 16, -1)
        qkv_106 = None
        qkv_107 = reshape_106.permute(2, 0, 3, 1, 4)
        reshape_106 = None
        unbind_53 = qkv_107.unbind(0)
        qkv_107 = None
        q_53 = unbind_53[0]
        k_53 = unbind_53[1]
        v_53 = unbind_53[2]
        unbind_53 = None
        x_694 = torch._C._nn.scaled_dot_product_attention(
            q_53, k_53, v_53, attn_mask=None, dropout_p=0.0
        )
        q_53 = k_53 = v_53 = None
        transpose_54 = x_694.transpose(1, 2)
        x_694 = None
        x_695 = transpose_54.reshape(1, 257, 1792)
        transpose_54 = None
        x_696 = torch._C._nn.linear(
            x_695,
            l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_bias_,
        )
        x_695 = l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_53_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_697 = torch.nn.functional.dropout(x_696, 0.0, False, False)
        x_696 = None
        x_698 = torch.nn.functional.layer_norm(
            x_697,
            (1792,),
            l_self_modules_blocks_modules_53_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_53_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_697 = (
            l_self_modules_blocks_modules_53_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_53_modules_norm1_parameters_bias_ = None
        x_699 = x_693 + x_698
        x_693 = x_698 = None
        x_700 = torch._C._nn.linear(
            x_699,
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_701 = torch._C._nn.gelu(x_700, approximate="none")
        x_700 = None
        x_702 = torch.nn.functional.dropout(x_701, 0.0, False, False)
        x_701 = None
        x_703 = torch._C._nn.linear(
            x_702,
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_702 = (
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_53_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_704 = torch.nn.functional.dropout(x_703, 0.0, False, False)
        x_703 = None
        x_705 = torch.nn.functional.layer_norm(
            x_704,
            (1792,),
            l_self_modules_blocks_modules_53_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_53_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_704 = (
            l_self_modules_blocks_modules_53_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_53_modules_norm2_parameters_bias_ = None
        x_706 = x_699 + x_705
        x_699 = x_705 = None
        qkv_bias_54 = torch.cat(
            (
                l_self_modules_blocks_modules_54_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_54_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_54_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_54_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_54_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_54_modules_attn_parameters_v_bias_ = None
        qkv_108 = torch._C._nn.linear(
            x_706,
            weight=l_self_modules_blocks_modules_54_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_54,
        )
        l_self_modules_blocks_modules_54_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_54
        ) = None
        reshape_108 = qkv_108.reshape(1, 257, 3, 16, -1)
        qkv_108 = None
        qkv_109 = reshape_108.permute(2, 0, 3, 1, 4)
        reshape_108 = None
        unbind_54 = qkv_109.unbind(0)
        qkv_109 = None
        q_54 = unbind_54[0]
        k_54 = unbind_54[1]
        v_54 = unbind_54[2]
        unbind_54 = None
        x_707 = torch._C._nn.scaled_dot_product_attention(
            q_54, k_54, v_54, attn_mask=None, dropout_p=0.0
        )
        q_54 = k_54 = v_54 = None
        transpose_55 = x_707.transpose(1, 2)
        x_707 = None
        x_708 = transpose_55.reshape(1, 257, 1792)
        transpose_55 = None
        x_709 = torch._C._nn.linear(
            x_708,
            l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_bias_,
        )
        x_708 = l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_54_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_710 = torch.nn.functional.dropout(x_709, 0.0, False, False)
        x_709 = None
        x_711 = torch.nn.functional.layer_norm(
            x_710,
            (1792,),
            l_self_modules_blocks_modules_54_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_54_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_710 = (
            l_self_modules_blocks_modules_54_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_54_modules_norm1_parameters_bias_ = None
        x_712 = x_706 + x_711
        x_706 = x_711 = None
        x_713 = torch._C._nn.linear(
            x_712,
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_714 = torch._C._nn.gelu(x_713, approximate="none")
        x_713 = None
        x_715 = torch.nn.functional.dropout(x_714, 0.0, False, False)
        x_714 = None
        x_716 = torch._C._nn.linear(
            x_715,
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_715 = (
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_54_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_717 = torch.nn.functional.dropout(x_716, 0.0, False, False)
        x_716 = None
        x_718 = torch.nn.functional.layer_norm(
            x_717,
            (1792,),
            l_self_modules_blocks_modules_54_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_54_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_717 = (
            l_self_modules_blocks_modules_54_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_54_modules_norm2_parameters_bias_ = None
        x_719 = x_712 + x_718
        x_712 = x_718 = None
        qkv_bias_55 = torch.cat(
            (
                l_self_modules_blocks_modules_55_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_55_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_55_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_55_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_55_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_55_modules_attn_parameters_v_bias_ = None
        qkv_110 = torch._C._nn.linear(
            x_719,
            weight=l_self_modules_blocks_modules_55_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_55,
        )
        l_self_modules_blocks_modules_55_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_55
        ) = None
        reshape_110 = qkv_110.reshape(1, 257, 3, 16, -1)
        qkv_110 = None
        qkv_111 = reshape_110.permute(2, 0, 3, 1, 4)
        reshape_110 = None
        unbind_55 = qkv_111.unbind(0)
        qkv_111 = None
        q_55 = unbind_55[0]
        k_55 = unbind_55[1]
        v_55 = unbind_55[2]
        unbind_55 = None
        x_720 = torch._C._nn.scaled_dot_product_attention(
            q_55, k_55, v_55, attn_mask=None, dropout_p=0.0
        )
        q_55 = k_55 = v_55 = None
        transpose_56 = x_720.transpose(1, 2)
        x_720 = None
        x_721 = transpose_56.reshape(1, 257, 1792)
        transpose_56 = None
        x_722 = torch._C._nn.linear(
            x_721,
            l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_bias_,
        )
        x_721 = l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_55_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_723 = torch.nn.functional.dropout(x_722, 0.0, False, False)
        x_722 = None
        x_724 = torch.nn.functional.layer_norm(
            x_723,
            (1792,),
            l_self_modules_blocks_modules_55_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_55_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_723 = (
            l_self_modules_blocks_modules_55_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_55_modules_norm1_parameters_bias_ = None
        x_725 = x_719 + x_724
        x_719 = x_724 = None
        x_726 = torch._C._nn.linear(
            x_725,
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_727 = torch._C._nn.gelu(x_726, approximate="none")
        x_726 = None
        x_728 = torch.nn.functional.dropout(x_727, 0.0, False, False)
        x_727 = None
        x_729 = torch._C._nn.linear(
            x_728,
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_728 = (
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_55_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_730 = torch.nn.functional.dropout(x_729, 0.0, False, False)
        x_729 = None
        x_731 = torch.nn.functional.layer_norm(
            x_730,
            (1792,),
            l_self_modules_blocks_modules_55_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_55_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_730 = (
            l_self_modules_blocks_modules_55_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_55_modules_norm2_parameters_bias_ = None
        x_732 = x_725 + x_731
        x_725 = x_731 = None
        qkv_bias_56 = torch.cat(
            (
                l_self_modules_blocks_modules_56_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_56_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_56_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_56_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_56_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_56_modules_attn_parameters_v_bias_ = None
        qkv_112 = torch._C._nn.linear(
            x_732,
            weight=l_self_modules_blocks_modules_56_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_56,
        )
        l_self_modules_blocks_modules_56_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_56
        ) = None
        reshape_112 = qkv_112.reshape(1, 257, 3, 16, -1)
        qkv_112 = None
        qkv_113 = reshape_112.permute(2, 0, 3, 1, 4)
        reshape_112 = None
        unbind_56 = qkv_113.unbind(0)
        qkv_113 = None
        q_56 = unbind_56[0]
        k_56 = unbind_56[1]
        v_56 = unbind_56[2]
        unbind_56 = None
        x_733 = torch._C._nn.scaled_dot_product_attention(
            q_56, k_56, v_56, attn_mask=None, dropout_p=0.0
        )
        q_56 = k_56 = v_56 = None
        transpose_57 = x_733.transpose(1, 2)
        x_733 = None
        x_734 = transpose_57.reshape(1, 257, 1792)
        transpose_57 = None
        x_735 = torch._C._nn.linear(
            x_734,
            l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_bias_,
        )
        x_734 = l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_56_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_736 = torch.nn.functional.dropout(x_735, 0.0, False, False)
        x_735 = None
        x_737 = torch.nn.functional.layer_norm(
            x_736,
            (1792,),
            l_self_modules_blocks_modules_56_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_56_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_736 = (
            l_self_modules_blocks_modules_56_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_56_modules_norm1_parameters_bias_ = None
        x_738 = x_732 + x_737
        x_732 = x_737 = None
        x_739 = torch._C._nn.linear(
            x_738,
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_740 = torch._C._nn.gelu(x_739, approximate="none")
        x_739 = None
        x_741 = torch.nn.functional.dropout(x_740, 0.0, False, False)
        x_740 = None
        x_742 = torch._C._nn.linear(
            x_741,
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_741 = (
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_56_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_743 = torch.nn.functional.dropout(x_742, 0.0, False, False)
        x_742 = None
        x_744 = torch.nn.functional.layer_norm(
            x_743,
            (1792,),
            l_self_modules_blocks_modules_56_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_56_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_743 = (
            l_self_modules_blocks_modules_56_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_56_modules_norm2_parameters_bias_ = None
        x_745 = x_738 + x_744
        x_738 = x_744 = None
        qkv_bias_57 = torch.cat(
            (
                l_self_modules_blocks_modules_57_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_57_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_57_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_57_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_57_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_57_modules_attn_parameters_v_bias_ = None
        qkv_114 = torch._C._nn.linear(
            x_745,
            weight=l_self_modules_blocks_modules_57_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_57,
        )
        l_self_modules_blocks_modules_57_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_57
        ) = None
        reshape_114 = qkv_114.reshape(1, 257, 3, 16, -1)
        qkv_114 = None
        qkv_115 = reshape_114.permute(2, 0, 3, 1, 4)
        reshape_114 = None
        unbind_57 = qkv_115.unbind(0)
        qkv_115 = None
        q_57 = unbind_57[0]
        k_57 = unbind_57[1]
        v_57 = unbind_57[2]
        unbind_57 = None
        x_746 = torch._C._nn.scaled_dot_product_attention(
            q_57, k_57, v_57, attn_mask=None, dropout_p=0.0
        )
        q_57 = k_57 = v_57 = None
        transpose_58 = x_746.transpose(1, 2)
        x_746 = None
        x_747 = transpose_58.reshape(1, 257, 1792)
        transpose_58 = None
        x_748 = torch._C._nn.linear(
            x_747,
            l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_bias_,
        )
        x_747 = l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_57_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_749 = torch.nn.functional.dropout(x_748, 0.0, False, False)
        x_748 = None
        x_750 = torch.nn.functional.layer_norm(
            x_749,
            (1792,),
            l_self_modules_blocks_modules_57_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_57_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_749 = (
            l_self_modules_blocks_modules_57_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_57_modules_norm1_parameters_bias_ = None
        x_751 = x_745 + x_750
        x_745 = x_750 = None
        x_752 = torch._C._nn.linear(
            x_751,
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_753 = torch._C._nn.gelu(x_752, approximate="none")
        x_752 = None
        x_754 = torch.nn.functional.dropout(x_753, 0.0, False, False)
        x_753 = None
        x_755 = torch._C._nn.linear(
            x_754,
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_754 = (
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_57_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_756 = torch.nn.functional.dropout(x_755, 0.0, False, False)
        x_755 = None
        x_757 = torch.nn.functional.layer_norm(
            x_756,
            (1792,),
            l_self_modules_blocks_modules_57_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_57_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_756 = (
            l_self_modules_blocks_modules_57_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_57_modules_norm2_parameters_bias_ = None
        x_758 = x_751 + x_757
        x_751 = x_757 = None
        qkv_bias_58 = torch.cat(
            (
                l_self_modules_blocks_modules_58_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_58_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_58_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_58_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_58_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_58_modules_attn_parameters_v_bias_ = None
        qkv_116 = torch._C._nn.linear(
            x_758,
            weight=l_self_modules_blocks_modules_58_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_58,
        )
        l_self_modules_blocks_modules_58_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_58
        ) = None
        reshape_116 = qkv_116.reshape(1, 257, 3, 16, -1)
        qkv_116 = None
        qkv_117 = reshape_116.permute(2, 0, 3, 1, 4)
        reshape_116 = None
        unbind_58 = qkv_117.unbind(0)
        qkv_117 = None
        q_58 = unbind_58[0]
        k_58 = unbind_58[1]
        v_58 = unbind_58[2]
        unbind_58 = None
        x_759 = torch._C._nn.scaled_dot_product_attention(
            q_58, k_58, v_58, attn_mask=None, dropout_p=0.0
        )
        q_58 = k_58 = v_58 = None
        transpose_59 = x_759.transpose(1, 2)
        x_759 = None
        x_760 = transpose_59.reshape(1, 257, 1792)
        transpose_59 = None
        x_761 = torch._C._nn.linear(
            x_760,
            l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_bias_,
        )
        x_760 = l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_58_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_762 = torch.nn.functional.dropout(x_761, 0.0, False, False)
        x_761 = None
        x_763 = torch.nn.functional.layer_norm(
            x_762,
            (1792,),
            l_self_modules_blocks_modules_58_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_58_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_762 = (
            l_self_modules_blocks_modules_58_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_58_modules_norm1_parameters_bias_ = None
        x_764 = x_758 + x_763
        x_758 = x_763 = None
        x_765 = torch._C._nn.linear(
            x_764,
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_766 = torch._C._nn.gelu(x_765, approximate="none")
        x_765 = None
        x_767 = torch.nn.functional.dropout(x_766, 0.0, False, False)
        x_766 = None
        x_768 = torch._C._nn.linear(
            x_767,
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_767 = (
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_58_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_769 = torch.nn.functional.dropout(x_768, 0.0, False, False)
        x_768 = None
        x_770 = torch.nn.functional.layer_norm(
            x_769,
            (1792,),
            l_self_modules_blocks_modules_58_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_58_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_769 = (
            l_self_modules_blocks_modules_58_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_58_modules_norm2_parameters_bias_ = None
        x_771 = x_764 + x_770
        x_764 = x_770 = None
        qkv_bias_59 = torch.cat(
            (
                l_self_modules_blocks_modules_59_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_59_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_59_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_59_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_59_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_59_modules_attn_parameters_v_bias_ = None
        qkv_118 = torch._C._nn.linear(
            x_771,
            weight=l_self_modules_blocks_modules_59_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_59,
        )
        l_self_modules_blocks_modules_59_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_59
        ) = None
        reshape_118 = qkv_118.reshape(1, 257, 3, 16, -1)
        qkv_118 = None
        qkv_119 = reshape_118.permute(2, 0, 3, 1, 4)
        reshape_118 = None
        unbind_59 = qkv_119.unbind(0)
        qkv_119 = None
        q_59 = unbind_59[0]
        k_59 = unbind_59[1]
        v_59 = unbind_59[2]
        unbind_59 = None
        x_772 = torch._C._nn.scaled_dot_product_attention(
            q_59, k_59, v_59, attn_mask=None, dropout_p=0.0
        )
        q_59 = k_59 = v_59 = None
        transpose_60 = x_772.transpose(1, 2)
        x_772 = None
        x_773 = transpose_60.reshape(1, 257, 1792)
        transpose_60 = None
        x_774 = torch._C._nn.linear(
            x_773,
            l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_bias_,
        )
        x_773 = l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_59_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_775 = torch.nn.functional.dropout(x_774, 0.0, False, False)
        x_774 = None
        x_776 = torch.nn.functional.layer_norm(
            x_775,
            (1792,),
            l_self_modules_blocks_modules_59_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_59_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_775 = (
            l_self_modules_blocks_modules_59_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_59_modules_norm1_parameters_bias_ = None
        x_777 = x_771 + x_776
        x_771 = x_776 = None
        x_778 = torch._C._nn.linear(
            x_777,
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_779 = torch._C._nn.gelu(x_778, approximate="none")
        x_778 = None
        x_780 = torch.nn.functional.dropout(x_779, 0.0, False, False)
        x_779 = None
        x_781 = torch._C._nn.linear(
            x_780,
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_780 = (
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_59_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_782 = torch.nn.functional.dropout(x_781, 0.0, False, False)
        x_781 = None
        x_783 = torch.nn.functional.layer_norm(
            x_782,
            (1792,),
            l_self_modules_blocks_modules_59_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_59_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_782 = (
            l_self_modules_blocks_modules_59_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_59_modules_norm2_parameters_bias_ = None
        x_784 = x_777 + x_783
        x_777 = x_783 = None
        qkv_bias_60 = torch.cat(
            (
                l_self_modules_blocks_modules_60_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_60_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_60_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_60_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_60_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_60_modules_attn_parameters_v_bias_ = None
        qkv_120 = torch._C._nn.linear(
            x_784,
            weight=l_self_modules_blocks_modules_60_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_60,
        )
        l_self_modules_blocks_modules_60_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_60
        ) = None
        reshape_120 = qkv_120.reshape(1, 257, 3, 16, -1)
        qkv_120 = None
        qkv_121 = reshape_120.permute(2, 0, 3, 1, 4)
        reshape_120 = None
        unbind_60 = qkv_121.unbind(0)
        qkv_121 = None
        q_60 = unbind_60[0]
        k_60 = unbind_60[1]
        v_60 = unbind_60[2]
        unbind_60 = None
        x_785 = torch._C._nn.scaled_dot_product_attention(
            q_60, k_60, v_60, attn_mask=None, dropout_p=0.0
        )
        q_60 = k_60 = v_60 = None
        transpose_61 = x_785.transpose(1, 2)
        x_785 = None
        x_786 = transpose_61.reshape(1, 257, 1792)
        transpose_61 = None
        x_787 = torch._C._nn.linear(
            x_786,
            l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_bias_,
        )
        x_786 = l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_60_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_788 = torch.nn.functional.dropout(x_787, 0.0, False, False)
        x_787 = None
        x_789 = torch.nn.functional.layer_norm(
            x_788,
            (1792,),
            l_self_modules_blocks_modules_60_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_60_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_788 = (
            l_self_modules_blocks_modules_60_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_60_modules_norm1_parameters_bias_ = None
        x_790 = x_784 + x_789
        x_784 = x_789 = None
        x_791 = torch._C._nn.linear(
            x_790,
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_792 = torch._C._nn.gelu(x_791, approximate="none")
        x_791 = None
        x_793 = torch.nn.functional.dropout(x_792, 0.0, False, False)
        x_792 = None
        x_794 = torch._C._nn.linear(
            x_793,
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_793 = (
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_60_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_795 = torch.nn.functional.dropout(x_794, 0.0, False, False)
        x_794 = None
        x_796 = torch.nn.functional.layer_norm(
            x_795,
            (1792,),
            l_self_modules_blocks_modules_60_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_60_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_795 = (
            l_self_modules_blocks_modules_60_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_60_modules_norm2_parameters_bias_ = None
        x_797 = x_790 + x_796
        x_790 = x_796 = None
        qkv_bias_61 = torch.cat(
            (
                l_self_modules_blocks_modules_61_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_61_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_61_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_61_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_61_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_61_modules_attn_parameters_v_bias_ = None
        qkv_122 = torch._C._nn.linear(
            x_797,
            weight=l_self_modules_blocks_modules_61_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_61,
        )
        l_self_modules_blocks_modules_61_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_61
        ) = None
        reshape_122 = qkv_122.reshape(1, 257, 3, 16, -1)
        qkv_122 = None
        qkv_123 = reshape_122.permute(2, 0, 3, 1, 4)
        reshape_122 = None
        unbind_61 = qkv_123.unbind(0)
        qkv_123 = None
        q_61 = unbind_61[0]
        k_61 = unbind_61[1]
        v_61 = unbind_61[2]
        unbind_61 = None
        x_798 = torch._C._nn.scaled_dot_product_attention(
            q_61, k_61, v_61, attn_mask=None, dropout_p=0.0
        )
        q_61 = k_61 = v_61 = None
        transpose_62 = x_798.transpose(1, 2)
        x_798 = None
        x_799 = transpose_62.reshape(1, 257, 1792)
        transpose_62 = None
        x_800 = torch._C._nn.linear(
            x_799,
            l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_bias_,
        )
        x_799 = l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_61_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_801 = torch.nn.functional.dropout(x_800, 0.0, False, False)
        x_800 = None
        x_802 = torch.nn.functional.layer_norm(
            x_801,
            (1792,),
            l_self_modules_blocks_modules_61_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_61_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_801 = (
            l_self_modules_blocks_modules_61_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_61_modules_norm1_parameters_bias_ = None
        x_803 = x_797 + x_802
        x_797 = x_802 = None
        x_804 = torch._C._nn.linear(
            x_803,
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_805 = torch._C._nn.gelu(x_804, approximate="none")
        x_804 = None
        x_806 = torch.nn.functional.dropout(x_805, 0.0, False, False)
        x_805 = None
        x_807 = torch._C._nn.linear(
            x_806,
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_806 = (
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_61_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_808 = torch.nn.functional.dropout(x_807, 0.0, False, False)
        x_807 = None
        x_809 = torch.nn.functional.layer_norm(
            x_808,
            (1792,),
            l_self_modules_blocks_modules_61_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_61_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_808 = (
            l_self_modules_blocks_modules_61_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_61_modules_norm2_parameters_bias_ = None
        x_810 = x_803 + x_809
        x_803 = x_809 = None
        qkv_bias_62 = torch.cat(
            (
                l_self_modules_blocks_modules_62_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_62_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_62_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_62_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_62_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_62_modules_attn_parameters_v_bias_ = None
        qkv_124 = torch._C._nn.linear(
            x_810,
            weight=l_self_modules_blocks_modules_62_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_62,
        )
        l_self_modules_blocks_modules_62_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_62
        ) = None
        reshape_124 = qkv_124.reshape(1, 257, 3, 16, -1)
        qkv_124 = None
        qkv_125 = reshape_124.permute(2, 0, 3, 1, 4)
        reshape_124 = None
        unbind_62 = qkv_125.unbind(0)
        qkv_125 = None
        q_62 = unbind_62[0]
        k_62 = unbind_62[1]
        v_62 = unbind_62[2]
        unbind_62 = None
        x_811 = torch._C._nn.scaled_dot_product_attention(
            q_62, k_62, v_62, attn_mask=None, dropout_p=0.0
        )
        q_62 = k_62 = v_62 = None
        transpose_63 = x_811.transpose(1, 2)
        x_811 = None
        x_812 = transpose_63.reshape(1, 257, 1792)
        transpose_63 = None
        x_813 = torch._C._nn.linear(
            x_812,
            l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_bias_,
        )
        x_812 = l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_62_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_814 = torch.nn.functional.dropout(x_813, 0.0, False, False)
        x_813 = None
        x_815 = torch.nn.functional.layer_norm(
            x_814,
            (1792,),
            l_self_modules_blocks_modules_62_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_62_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_814 = (
            l_self_modules_blocks_modules_62_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_62_modules_norm1_parameters_bias_ = None
        x_816 = x_810 + x_815
        x_810 = x_815 = None
        x_817 = torch._C._nn.linear(
            x_816,
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_818 = torch._C._nn.gelu(x_817, approximate="none")
        x_817 = None
        x_819 = torch.nn.functional.dropout(x_818, 0.0, False, False)
        x_818 = None
        x_820 = torch._C._nn.linear(
            x_819,
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_819 = (
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_62_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_821 = torch.nn.functional.dropout(x_820, 0.0, False, False)
        x_820 = None
        x_822 = torch.nn.functional.layer_norm(
            x_821,
            (1792,),
            l_self_modules_blocks_modules_62_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_62_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_821 = (
            l_self_modules_blocks_modules_62_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_62_modules_norm2_parameters_bias_ = None
        x_823 = x_816 + x_822
        x_816 = x_822 = None
        qkv_bias_63 = torch.cat(
            (
                l_self_modules_blocks_modules_63_modules_attn_parameters_q_bias_,
                l_self_modules_blocks_modules_63_modules_attn_buffers_k_bias_,
                l_self_modules_blocks_modules_63_modules_attn_parameters_v_bias_,
            )
        )
        l_self_modules_blocks_modules_63_modules_attn_parameters_q_bias_ = (
            l_self_modules_blocks_modules_63_modules_attn_buffers_k_bias_
        ) = l_self_modules_blocks_modules_63_modules_attn_parameters_v_bias_ = None
        qkv_126 = torch._C._nn.linear(
            x_823,
            weight=l_self_modules_blocks_modules_63_modules_attn_modules_qkv_parameters_weight_,
            bias=qkv_bias_63,
        )
        l_self_modules_blocks_modules_63_modules_attn_modules_qkv_parameters_weight_ = (
            qkv_bias_63
        ) = None
        reshape_126 = qkv_126.reshape(1, 257, 3, 16, -1)
        qkv_126 = None
        qkv_127 = reshape_126.permute(2, 0, 3, 1, 4)
        reshape_126 = None
        unbind_63 = qkv_127.unbind(0)
        qkv_127 = None
        q_63 = unbind_63[0]
        k_63 = unbind_63[1]
        v_63 = unbind_63[2]
        unbind_63 = None
        x_824 = torch._C._nn.scaled_dot_product_attention(
            q_63, k_63, v_63, attn_mask=None, dropout_p=0.0
        )
        q_63 = k_63 = v_63 = None
        transpose_64 = x_824.transpose(1, 2)
        x_824 = None
        x_825 = transpose_64.reshape(1, 257, 1792)
        transpose_64 = None
        x_826 = torch._C._nn.linear(
            x_825,
            l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_weight_,
            l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_bias_,
        )
        x_825 = l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_weight_ = (
            l_self_modules_blocks_modules_63_modules_attn_modules_proj_parameters_bias_
        ) = None
        x_827 = torch.nn.functional.dropout(x_826, 0.0, False, False)
        x_826 = None
        x_828 = torch.nn.functional.layer_norm(
            x_827,
            (1792,),
            l_self_modules_blocks_modules_63_modules_norm1_parameters_weight_,
            l_self_modules_blocks_modules_63_modules_norm1_parameters_bias_,
            1e-06,
        )
        x_827 = (
            l_self_modules_blocks_modules_63_modules_norm1_parameters_weight_
        ) = l_self_modules_blocks_modules_63_modules_norm1_parameters_bias_ = None
        x_829 = x_823 + x_828
        x_823 = x_828 = None
        x_830 = torch._C._nn.linear(
            x_829,
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_weight_,
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_bias_,
        )
        l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_weight_ = (
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc1_parameters_bias_
        ) = None
        x_831 = torch._C._nn.gelu(x_830, approximate="none")
        x_830 = None
        x_832 = torch.nn.functional.dropout(x_831, 0.0, False, False)
        x_831 = None
        x_833 = torch._C._nn.linear(
            x_832,
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_weight_,
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_bias_,
        )
        x_832 = (
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_63_modules_mlp_modules_fc2_parameters_bias_
        ) = None
        x_834 = torch.nn.functional.dropout(x_833, 0.0, False, False)
        x_833 = None
        x_835 = torch.nn.functional.layer_norm(
            x_834,
            (1792,),
            l_self_modules_blocks_modules_63_modules_norm2_parameters_weight_,
            l_self_modules_blocks_modules_63_modules_norm2_parameters_bias_,
            1e-06,
        )
        x_834 = (
            l_self_modules_blocks_modules_63_modules_norm2_parameters_weight_
        ) = l_self_modules_blocks_modules_63_modules_norm2_parameters_bias_ = None
        x_836 = x_829 + x_835
        x_829 = x_835 = None
        x_837 = torch.nn.functional.layer_norm(
            x_836,
            (1792,),
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            1e-06,
        )
        x_836 = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_838 = x_837[(slice(None, None, None), 0)]
        x_837 = None
        x_839 = torch.nn.functional.dropout(x_838, 0.0, False, False)
        x_838 = None
        x_840 = torch._C._nn.linear(
            x_839,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_839 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_840,)
