import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_pre_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_post_self_attn_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_embed_tokens_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_layers_modules_0_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_pre_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_pre_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_post_self_attn_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_post_self_attn_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_embed_tokens_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = l_self_modules_embed_tokens_parameters_weight_ = None
        cache_position = torch.arange(0, 10, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        kv_arange = torch.arange(10, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cuda", index=0))
        head_arange = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._C._functorch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._C._functorch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        result_1 = child_2.new_zeros((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        to_1 = le.to(device(type="cuda", index=0))
        le = None
        result_2 = result_1.__or__(to_1)
        result_1 = to_1 = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(l_kwargs_attention_mask_, [child, child_3])
        to_2 = index.to(torch.bool)
        index = None
        to_3 = to_2.to(device(type="cuda", index=0))
        to_2 = None
        result_3 = result_2.__or__(to_3)
        result_2 = to_3 = None
        to_4 = result_3.to(device(type="cuda", index=0))
        result_3 = None
        result_4 = result.__and__(to_4)
        result = to_4 = None
        function_ctx_1 = torch.autograd.function.FunctionCtx()
        function_ctx_1 = None
        index_1 = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        to_5 = index_1.to(device(type="cuda", index=0))
        index_1 = None
        result_5 = result_4.__and__(to_5)
        result_4 = to_5 = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_5, 4, 10, 0)
        result_5 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 10, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._C._functorch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._C._functorch._remove_batch_dim(batched_outputs_2, 1, 1, 0)
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_3 = None
        attention_mask_1 = l_kwargs_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        kv_arange_2 = torch.arange(10, device=device(type="cuda", index=0))
        kv_arange_2 += 0
        kv_arange_3 = kv_arange_2
        kv_arange_2 = None
        batch_arange_1 = torch.arange(1, device=device(type="cuda", index=0))
        head_arange_1 = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions_4 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_4 = None
        _vmap_increment_nesting_4 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_4 = None
        child_4 = torch._C._functorch._add_batch_dim(batch_arange_1, 0, 1)
        batch_arange_1 = None
        lazy_load_decompositions_5 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_5 = None
        _vmap_increment_nesting_5 = torch._C._functorch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting_5 = None
        child_5 = torch._C._functorch._add_batch_dim(head_arange_1, 0, 2)
        head_arange_1 = child_5 = None
        lazy_load_decompositions_6 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_6 = None
        _vmap_increment_nesting_6 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_6 = None
        child_6 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_7 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_7 = None
        _vmap_increment_nesting_7 = torch._C._functorch._vmap_increment_nesting(
            10, "error"
        )
        _vmap_increment_nesting_7 = None
        child_7 = torch._C._functorch._add_batch_dim(kv_arange_3, 0, 4)
        kv_arange_3 = None
        result_6 = child_6.new_ones((), dtype=torch.bool)
        result_7 = child_6.new_ones((), dtype=torch.bool)
        result_8 = child_6.new_zeros((), dtype=torch.bool)
        result_9 = child_6.new_ones((), dtype=torch.bool)
        sub = child_6.sub(4096)
        gt = child_7.gt(sub)
        sub = None
        to_7 = gt.to(device(type="cuda", index=0))
        gt = None
        result_10 = result_9.__and__(to_7)
        result_9 = to_7 = None
        le_1 = child_7.le(child_6)
        to_8 = le_1.to(device(type="cuda", index=0))
        le_1 = None
        result_11 = result_10.__and__(to_8)
        result_10 = to_8 = None
        to_9 = result_11.to(device(type="cuda", index=0))
        result_11 = None
        result_12 = result_8.__or__(to_9)
        result_8 = to_9 = None
        sub_1 = child_6.sub(4096)
        lt = sub_1.lt(child_7)
        sub_1 = None
        add = child_6.add(4096)
        child_6 = None
        lt_1 = child_7.lt(add)
        add = None
        and_5 = lt.__and__(lt_1)
        lt = lt_1 = None
        to_10 = and_5.to(device(type="cuda", index=0))
        and_5 = None
        result_13 = result_12.__or__(to_10)
        result_12 = to_10 = None
        to_11 = result_13.to(device(type="cuda", index=0))
        result_13 = None
        result_14 = result_7.__and__(to_11)
        result_7 = to_11 = None
        function_ctx_2 = torch.autograd.function.FunctionCtx()
        function_ctx_2 = None
        index_2 = torch.ops.aten.index(l_kwargs_attention_mask_, [child_4, child_7])
        l_kwargs_attention_mask_ = None
        to_12 = index_2.to(torch.bool)
        index_2 = None
        to_13 = to_12.to(device(type="cuda", index=0))
        to_12 = None
        result_15 = result_14.__and__(to_13)
        result_14 = to_13 = None
        to_14 = result_15.to(device(type="cuda", index=0))
        result_15 = None
        result_16 = result_6.__and__(to_14)
        result_6 = to_14 = None
        function_ctx_3 = torch.autograd.function.FunctionCtx()
        function_ctx_3 = None
        index_3 = torch.ops.aten.index(attention_mask_1, [child_4, child_7])
        attention_mask_1 = child_4 = child_7 = None
        to_15 = index_3.to(device(type="cuda", index=0))
        index_3 = None
        result_17 = result_16.__and__(to_15)
        result_16 = to_15 = None
        batched_outputs_3 = torch._C._functorch._remove_batch_dim(result_17, 4, 10, 0)
        result_17 = None
        _vmap_decrement_nesting_4 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_4 = None
        batched_outputs_4 = torch._C._functorch._remove_batch_dim(
            batched_outputs_3, 3, 10, 0
        )
        batched_outputs_3 = None
        _vmap_decrement_nesting_5 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_5 = None
        batched_outputs_5 = torch._C._functorch._remove_batch_dim(
            batched_outputs_4, 2, 1, 0
        )
        batched_outputs_4 = None
        _vmap_decrement_nesting_6 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_6 = None
        causal_mask_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs_5, 1, 1, 0
        )
        batched_outputs_5 = None
        _vmap_decrement_nesting_7 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_7 = None
        _set_grad_enabled = torch._C._set_grad_enabled(False)
        _set_grad_enabled = None
        getitem = l_self_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem.float()
        getitem = None
        expand = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand.to(device(type="cuda", index=0))
        expand = None
        getitem_1 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_1.float()
        getitem_1 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast(
            "cuda", None, False, None
        )
        float_3 = inv_freq_expanded.float()
        inv_freq_expanded = None
        float_4 = position_ids_expanded.float()
        position_ids_expanded = None
        matmul = float_3 @ float_4
        float_3 = float_4 = None
        freqs = matmul.transpose(1, 2)
        matmul = None
        emb = torch.cat((freqs, freqs), dim=-1)
        freqs = None
        cos = emb.cos()
        cos_1 = cos * 1.0
        cos = None
        sin = emb.sin()
        emb = None
        sin_1 = sin * 1.0
        sin = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast)
        _enter_autocast = _exit_autocast = None
        cos_2 = cos_1.to(dtype=torch.float32)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float32)
        sin_1 = None
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        _set_grad_enabled_1 = None
        normalizer = torch.tensor(32.0, dtype=torch.float32)
        hidden_states = inputs_embeds * normalizer
        inputs_embeds = normalizer = None
        hidden_states_1 = torch.nn.functional.dropout(hidden_states, 0.0, False, False)
        hidden_states = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        float_5 = hidden_states_1.float()
        pow_1 = float_5.pow(2)
        mean = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add_1 = mean + 1e-06
        mean = None
        rsqrt = torch.rsqrt(add_1)
        add_1 = None
        output = float_5 * rsqrt
        float_5 = rsqrt = None
        float_6 = (
            l_self_modules_layers_modules_0_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_2 = 1.0 + float_6
        float_6 = None
        output_1 = output * add_2
        output = add_2 = None
        hidden_states_2 = output_1.type_as(hidden_states_1)
        output_1 = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 10, -1, 64))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 10, -1, 64))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = linear_2.view((1, 10, -1, 64))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_5 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 32, None))]
        x2 = query_states[(Ellipsis, slice(32, None, None))]
        query_states = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_6 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_5 + mul_6
        mul_5 = mul_6 = None
        mul_7 = key_states * cos_3
        cos_3 = None
        x1_1 = key_states[(Ellipsis, slice(None, 32, None))]
        x2_1 = key_states[(Ellipsis, slice(32, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_8 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_7 + mul_8
        mul_7 = mul_8 = None
        attention_mask_2 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query = q_embed.contiguous()
        q_embed = None
        key = k_embed.contiguous()
        k_embed = None
        value = value_states.contiguous()
        value_states = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query = key = value = attention_mask_2 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape = attn_output_1.reshape(1, 10, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_7 = attn_output_3.float()
        pow_2 = float_7.pow(2)
        mean_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_5 = mean_1 + 1e-06
        mean_1 = None
        rsqrt_1 = torch.rsqrt(add_5)
        add_5 = None
        output_2 = float_7 * rsqrt_1
        float_7 = rsqrt_1 = None
        float_8 = (
            l_self_modules_layers_modules_0_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_6 = 1.0 + float_8
        float_8 = None
        output_3 = output_2 * add_6
        output_2 = add_6 = None
        hidden_states_3 = output_3.type_as(attn_output_3)
        output_3 = attn_output_3 = None
        dropout_1 = torch.nn.functional.dropout(hidden_states_3, 0.0, False, False)
        hidden_states_3 = None
        hidden_states_4 = hidden_states_1 + dropout_1
        hidden_states_1 = dropout_1 = None
        float_9 = hidden_states_4.float()
        pow_3 = float_9.pow(2)
        mean_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_8 = mean_2 + 1e-06
        mean_2 = None
        rsqrt_2 = torch.rsqrt(add_8)
        add_8 = None
        output_4 = float_9 * rsqrt_2
        float_9 = rsqrt_2 = None
        float_10 = (
            l_self_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_9 = 1.0 + float_10
        float_10 = None
        output_5 = output_4 * add_9
        output_4 = add_9 = None
        hidden_states_5 = output_5.type_as(hidden_states_4)
        output_5 = None
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu = torch._C._nn.gelu(linear_4, approximate="tanh")
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_5 = l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_6 = gelu * linear_5
        gelu = linear_5 = None
        hidden_states_7 = torch.nn.functional.dropout(
            hidden_states_6, 0.0, False, False
        )
        hidden_states_6 = None
        down_proj = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_7 = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_11 = down_proj.float()
        pow_4 = float_11.pow(2)
        mean_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = mean_3 + 1e-06
        mean_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        output_6 = float_11 * rsqrt_3
        float_11 = rsqrt_3 = None
        float_12 = (
            l_self_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_11 = 1.0 + float_12
        float_12 = None
        output_7 = output_6 * add_11
        output_6 = add_11 = None
        hidden_states_8 = output_7.type_as(down_proj)
        output_7 = down_proj = None
        dropout_3 = torch.nn.functional.dropout(hidden_states_8, 0.0, False, False)
        hidden_states_8 = None
        hidden_states_9 = hidden_states_4 + dropout_3
        hidden_states_4 = dropout_3 = None
        float_13 = hidden_states_9.float()
        pow_5 = float_13.pow(2)
        mean_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_13 = mean_4 + 1e-06
        mean_4 = None
        rsqrt_4 = torch.rsqrt(add_13)
        add_13 = None
        output_8 = float_13 * rsqrt_4
        float_13 = rsqrt_4 = None
        float_14 = (
            l_self_modules_layers_modules_1_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_14 = 1.0 + float_14
        float_14 = None
        output_9 = output_8 * add_14
        output_8 = add_14 = None
        hidden_states_10 = output_9.type_as(hidden_states_9)
        output_9 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 10, -1, 64))
        linear_7 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 10, -1, 64))
        linear_8 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_10 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_9.view((1, 10, -1, 64))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_18 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 32, None))]
        x2_2 = query_states_1[(Ellipsis, slice(32, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_19 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_18 + mul_19
        mul_18 = mul_19 = None
        mul_20 = key_states_1 * cos_4
        cos_4 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 32, None))]
        x2_3 = key_states_1[(Ellipsis, slice(32, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_21 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_20 + mul_21
        mul_20 = mul_21 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_1 = q_embed_1.contiguous()
        q_embed_1 = None
        key_1 = k_embed_1.contiguous()
        k_embed_1 = None
        value_1 = value_states_1.contiguous()
        value_states_1 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_3 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_1 = attn_output_5.reshape(1, 10, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_15 = attn_output_7.float()
        pow_6 = float_15.pow(2)
        mean_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_17 = mean_5 + 1e-06
        mean_5 = None
        rsqrt_5 = torch.rsqrt(add_17)
        add_17 = None
        output_10 = float_15 * rsqrt_5
        float_15 = rsqrt_5 = None
        float_16 = (
            l_self_modules_layers_modules_1_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_18 = 1.0 + float_16
        float_16 = None
        output_11 = output_10 * add_18
        output_10 = add_18 = None
        hidden_states_11 = output_11.type_as(attn_output_7)
        output_11 = attn_output_7 = None
        dropout_4 = torch.nn.functional.dropout(hidden_states_11, 0.0, False, False)
        hidden_states_11 = None
        hidden_states_12 = hidden_states_9 + dropout_4
        hidden_states_9 = dropout_4 = None
        float_17 = hidden_states_12.float()
        pow_7 = float_17.pow(2)
        mean_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_20 = mean_6 + 1e-06
        mean_6 = None
        rsqrt_6 = torch.rsqrt(add_20)
        add_20 = None
        output_12 = float_17 * rsqrt_6
        float_17 = rsqrt_6 = None
        float_18 = (
            l_self_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_21 = 1.0 + float_18
        float_18 = None
        output_13 = output_12 * add_21
        output_12 = add_21 = None
        hidden_states_13 = output_13.type_as(hidden_states_12)
        output_13 = None
        linear_11 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_1 = torch._C._nn.gelu(linear_11, approximate="tanh")
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_13 = l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_14 = gelu_1 * linear_12
        gelu_1 = linear_12 = None
        hidden_states_15 = torch.nn.functional.dropout(
            hidden_states_14, 0.0, False, False
        )
        hidden_states_14 = None
        down_proj_1 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_15 = l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_19 = down_proj_1.float()
        pow_8 = float_19.pow(2)
        mean_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_22 = mean_7 + 1e-06
        mean_7 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        output_14 = float_19 * rsqrt_7
        float_19 = rsqrt_7 = None
        float_20 = (
            l_self_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_23 = 1.0 + float_20
        float_20 = None
        output_15 = output_14 * add_23
        output_14 = add_23 = None
        hidden_states_16 = output_15.type_as(down_proj_1)
        output_15 = down_proj_1 = None
        dropout_6 = torch.nn.functional.dropout(hidden_states_16, 0.0, False, False)
        hidden_states_16 = None
        hidden_states_17 = hidden_states_12 + dropout_6
        hidden_states_12 = dropout_6 = None
        float_21 = hidden_states_17.float()
        pow_9 = float_21.pow(2)
        mean_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_25 = mean_8 + 1e-06
        mean_8 = None
        rsqrt_8 = torch.rsqrt(add_25)
        add_25 = None
        output_16 = float_21 * rsqrt_8
        float_21 = rsqrt_8 = None
        float_22 = (
            l_self_modules_layers_modules_2_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_26 = 1.0 + float_22
        float_22 = None
        output_17 = output_16 * add_26
        output_16 = add_26 = None
        hidden_states_18 = output_17.type_as(hidden_states_17)
        output_17 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 10, -1, 64))
        linear_14 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 10, -1, 64))
        linear_15 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_18 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_16.view((1, 10, -1, 64))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_31 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 32, None))]
        x2_4 = query_states_2[(Ellipsis, slice(32, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_32 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_31 + mul_32
        mul_31 = mul_32 = None
        mul_33 = key_states_2 * cos_5
        cos_5 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 32, None))]
        x2_5 = key_states_2[(Ellipsis, slice(32, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_34 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_33 + mul_34
        mul_33 = mul_34 = None
        attention_mask_4 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_2 = q_embed_2.contiguous()
        q_embed_2 = None
        key_2 = k_embed_2.contiguous()
        k_embed_2 = None
        value_2 = value_states_2.contiguous()
        value_states_2 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_4 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_2 = attn_output_9.reshape(1, 10, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_23 = attn_output_11.float()
        pow_10 = float_23.pow(2)
        mean_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_29 = mean_9 + 1e-06
        mean_9 = None
        rsqrt_9 = torch.rsqrt(add_29)
        add_29 = None
        output_18 = float_23 * rsqrt_9
        float_23 = rsqrt_9 = None
        float_24 = (
            l_self_modules_layers_modules_2_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_30 = 1.0 + float_24
        float_24 = None
        output_19 = output_18 * add_30
        output_18 = add_30 = None
        hidden_states_19 = output_19.type_as(attn_output_11)
        output_19 = attn_output_11 = None
        dropout_7 = torch.nn.functional.dropout(hidden_states_19, 0.0, False, False)
        hidden_states_19 = None
        hidden_states_20 = hidden_states_17 + dropout_7
        hidden_states_17 = dropout_7 = None
        float_25 = hidden_states_20.float()
        pow_11 = float_25.pow(2)
        mean_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_32 = mean_10 + 1e-06
        mean_10 = None
        rsqrt_10 = torch.rsqrt(add_32)
        add_32 = None
        output_20 = float_25 * rsqrt_10
        float_25 = rsqrt_10 = None
        float_26 = (
            l_self_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_33 = 1.0 + float_26
        float_26 = None
        output_21 = output_20 * add_33
        output_20 = add_33 = None
        hidden_states_21 = output_21.type_as(hidden_states_20)
        output_21 = None
        linear_18 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_2 = torch._C._nn.gelu(linear_18, approximate="tanh")
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_21 = l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_22 = gelu_2 * linear_19
        gelu_2 = linear_19 = None
        hidden_states_23 = torch.nn.functional.dropout(
            hidden_states_22, 0.0, False, False
        )
        hidden_states_22 = None
        down_proj_2 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_23 = l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_27 = down_proj_2.float()
        pow_12 = float_27.pow(2)
        mean_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_34 = mean_11 + 1e-06
        mean_11 = None
        rsqrt_11 = torch.rsqrt(add_34)
        add_34 = None
        output_22 = float_27 * rsqrt_11
        float_27 = rsqrt_11 = None
        float_28 = (
            l_self_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_35 = 1.0 + float_28
        float_28 = None
        output_23 = output_22 * add_35
        output_22 = add_35 = None
        hidden_states_24 = output_23.type_as(down_proj_2)
        output_23 = down_proj_2 = None
        dropout_9 = torch.nn.functional.dropout(hidden_states_24, 0.0, False, False)
        hidden_states_24 = None
        hidden_states_25 = hidden_states_20 + dropout_9
        hidden_states_20 = dropout_9 = None
        float_29 = hidden_states_25.float()
        pow_13 = float_29.pow(2)
        mean_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_37 = mean_12 + 1e-06
        mean_12 = None
        rsqrt_12 = torch.rsqrt(add_37)
        add_37 = None
        output_24 = float_29 * rsqrt_12
        float_29 = rsqrt_12 = None
        float_30 = (
            l_self_modules_layers_modules_3_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_38 = 1.0 + float_30
        float_30 = None
        output_25 = output_24 * add_38
        output_24 = add_38 = None
        hidden_states_26 = output_25.type_as(hidden_states_25)
        output_25 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 10, -1, 64))
        linear_21 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 10, -1, 64))
        linear_22 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_26 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_23.view((1, 10, -1, 64))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_44 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 32, None))]
        x2_6 = query_states_3[(Ellipsis, slice(32, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_45 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_44 + mul_45
        mul_44 = mul_45 = None
        mul_46 = key_states_3 * cos_6
        cos_6 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 32, None))]
        x2_7 = key_states_3[(Ellipsis, slice(32, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_47 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_46 + mul_47
        mul_46 = mul_47 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_3 = q_embed_3.contiguous()
        q_embed_3 = None
        key_3 = k_embed_3.contiguous()
        k_embed_3 = None
        value_3 = value_states_3.contiguous()
        value_states_3 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_5 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_3 = attn_output_13.reshape(1, 10, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_31 = attn_output_15.float()
        pow_14 = float_31.pow(2)
        mean_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_41 = mean_13 + 1e-06
        mean_13 = None
        rsqrt_13 = torch.rsqrt(add_41)
        add_41 = None
        output_26 = float_31 * rsqrt_13
        float_31 = rsqrt_13 = None
        float_32 = (
            l_self_modules_layers_modules_3_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_42 = 1.0 + float_32
        float_32 = None
        output_27 = output_26 * add_42
        output_26 = add_42 = None
        hidden_states_27 = output_27.type_as(attn_output_15)
        output_27 = attn_output_15 = None
        dropout_10 = torch.nn.functional.dropout(hidden_states_27, 0.0, False, False)
        hidden_states_27 = None
        hidden_states_28 = hidden_states_25 + dropout_10
        hidden_states_25 = dropout_10 = None
        float_33 = hidden_states_28.float()
        pow_15 = float_33.pow(2)
        mean_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_44 = mean_14 + 1e-06
        mean_14 = None
        rsqrt_14 = torch.rsqrt(add_44)
        add_44 = None
        output_28 = float_33 * rsqrt_14
        float_33 = rsqrt_14 = None
        float_34 = (
            l_self_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_45 = 1.0 + float_34
        float_34 = None
        output_29 = output_28 * add_45
        output_28 = add_45 = None
        hidden_states_29 = output_29.type_as(hidden_states_28)
        output_29 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_3 = torch._C._nn.gelu(linear_25, approximate="tanh")
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_29 = l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_30 = gelu_3 * linear_26
        gelu_3 = linear_26 = None
        hidden_states_31 = torch.nn.functional.dropout(
            hidden_states_30, 0.0, False, False
        )
        hidden_states_30 = None
        down_proj_3 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_31 = l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_35 = down_proj_3.float()
        pow_16 = float_35.pow(2)
        mean_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_46 = mean_15 + 1e-06
        mean_15 = None
        rsqrt_15 = torch.rsqrt(add_46)
        add_46 = None
        output_30 = float_35 * rsqrt_15
        float_35 = rsqrt_15 = None
        float_36 = (
            l_self_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_47 = 1.0 + float_36
        float_36 = None
        output_31 = output_30 * add_47
        output_30 = add_47 = None
        hidden_states_32 = output_31.type_as(down_proj_3)
        output_31 = down_proj_3 = None
        dropout_12 = torch.nn.functional.dropout(hidden_states_32, 0.0, False, False)
        hidden_states_32 = None
        hidden_states_33 = hidden_states_28 + dropout_12
        hidden_states_28 = dropout_12 = None
        float_37 = hidden_states_33.float()
        pow_17 = float_37.pow(2)
        mean_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_49 = mean_16 + 1e-06
        mean_16 = None
        rsqrt_16 = torch.rsqrt(add_49)
        add_49 = None
        output_32 = float_37 * rsqrt_16
        float_37 = rsqrt_16 = None
        float_38 = (
            l_self_modules_layers_modules_4_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_50 = 1.0 + float_38
        float_38 = None
        output_33 = output_32 * add_50
        output_32 = add_50 = None
        hidden_states_34 = output_33.type_as(hidden_states_33)
        output_33 = None
        linear_28 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 10, -1, 64))
        linear_28 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 10, -1, 64))
        linear_29 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_34,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_34 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_30.view((1, 10, -1, 64))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_57 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 32, None))]
        x2_8 = query_states_4[(Ellipsis, slice(32, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_58 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_57 + mul_58
        mul_57 = mul_58 = None
        mul_59 = key_states_4 * cos_7
        cos_7 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 32, None))]
        x2_9 = key_states_4[(Ellipsis, slice(32, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_60 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_59 + mul_60
        mul_59 = mul_60 = None
        attention_mask_6 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_4 = q_embed_4.contiguous()
        q_embed_4 = None
        key_4 = k_embed_4.contiguous()
        k_embed_4 = None
        value_4 = value_states_4.contiguous()
        value_states_4 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_6 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_4 = attn_output_17.reshape(1, 10, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_39 = attn_output_19.float()
        pow_18 = float_39.pow(2)
        mean_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_53 = mean_17 + 1e-06
        mean_17 = None
        rsqrt_17 = torch.rsqrt(add_53)
        add_53 = None
        output_34 = float_39 * rsqrt_17
        float_39 = rsqrt_17 = None
        float_40 = (
            l_self_modules_layers_modules_4_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_54 = 1.0 + float_40
        float_40 = None
        output_35 = output_34 * add_54
        output_34 = add_54 = None
        hidden_states_35 = output_35.type_as(attn_output_19)
        output_35 = attn_output_19 = None
        dropout_13 = torch.nn.functional.dropout(hidden_states_35, 0.0, False, False)
        hidden_states_35 = None
        hidden_states_36 = hidden_states_33 + dropout_13
        hidden_states_33 = dropout_13 = None
        float_41 = hidden_states_36.float()
        pow_19 = float_41.pow(2)
        mean_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_56 = mean_18 + 1e-06
        mean_18 = None
        rsqrt_18 = torch.rsqrt(add_56)
        add_56 = None
        output_36 = float_41 * rsqrt_18
        float_41 = rsqrt_18 = None
        float_42 = (
            l_self_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_57 = 1.0 + float_42
        float_42 = None
        output_37 = output_36 * add_57
        output_36 = add_57 = None
        hidden_states_37 = output_37.type_as(hidden_states_36)
        output_37 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_4 = torch._C._nn.gelu(linear_32, approximate="tanh")
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_37 = l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_38 = gelu_4 * linear_33
        gelu_4 = linear_33 = None
        hidden_states_39 = torch.nn.functional.dropout(
            hidden_states_38, 0.0, False, False
        )
        hidden_states_38 = None
        down_proj_4 = torch._C._nn.linear(
            hidden_states_39,
            l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_39 = l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_43 = down_proj_4.float()
        pow_20 = float_43.pow(2)
        mean_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_58 = mean_19 + 1e-06
        mean_19 = None
        rsqrt_19 = torch.rsqrt(add_58)
        add_58 = None
        output_38 = float_43 * rsqrt_19
        float_43 = rsqrt_19 = None
        float_44 = (
            l_self_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_59 = 1.0 + float_44
        float_44 = None
        output_39 = output_38 * add_59
        output_38 = add_59 = None
        hidden_states_40 = output_39.type_as(down_proj_4)
        output_39 = down_proj_4 = None
        dropout_15 = torch.nn.functional.dropout(hidden_states_40, 0.0, False, False)
        hidden_states_40 = None
        hidden_states_41 = hidden_states_36 + dropout_15
        hidden_states_36 = dropout_15 = None
        float_45 = hidden_states_41.float()
        pow_21 = float_45.pow(2)
        mean_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_61 = mean_20 + 1e-06
        mean_20 = None
        rsqrt_20 = torch.rsqrt(add_61)
        add_61 = None
        output_40 = float_45 * rsqrt_20
        float_45 = rsqrt_20 = None
        float_46 = (
            l_self_modules_layers_modules_5_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_62 = 1.0 + float_46
        float_46 = None
        output_41 = output_40 * add_62
        output_40 = add_62 = None
        hidden_states_42 = output_41.type_as(hidden_states_41)
        output_41 = None
        linear_35 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 10, -1, 64))
        linear_35 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 10, -1, 64))
        linear_36 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_37.view((1, 10, -1, 64))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_70 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 32, None))]
        x2_10 = query_states_5[(Ellipsis, slice(32, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_71 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_70 + mul_71
        mul_70 = mul_71 = None
        mul_72 = key_states_5 * cos_8
        cos_8 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 32, None))]
        x2_11 = key_states_5[(Ellipsis, slice(32, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_73 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_72 + mul_73
        mul_72 = mul_73 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_5 = q_embed_5.contiguous()
        q_embed_5 = None
        key_5 = k_embed_5.contiguous()
        k_embed_5 = None
        value_5 = value_states_5.contiguous()
        value_states_5 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_7 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_5 = attn_output_21.reshape(1, 10, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_47 = attn_output_23.float()
        pow_22 = float_47.pow(2)
        mean_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_65 = mean_21 + 1e-06
        mean_21 = None
        rsqrt_21 = torch.rsqrt(add_65)
        add_65 = None
        output_42 = float_47 * rsqrt_21
        float_47 = rsqrt_21 = None
        float_48 = (
            l_self_modules_layers_modules_5_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_66 = 1.0 + float_48
        float_48 = None
        output_43 = output_42 * add_66
        output_42 = add_66 = None
        hidden_states_43 = output_43.type_as(attn_output_23)
        output_43 = attn_output_23 = None
        dropout_16 = torch.nn.functional.dropout(hidden_states_43, 0.0, False, False)
        hidden_states_43 = None
        hidden_states_44 = hidden_states_41 + dropout_16
        hidden_states_41 = dropout_16 = None
        float_49 = hidden_states_44.float()
        pow_23 = float_49.pow(2)
        mean_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_68 = mean_22 + 1e-06
        mean_22 = None
        rsqrt_22 = torch.rsqrt(add_68)
        add_68 = None
        output_44 = float_49 * rsqrt_22
        float_49 = rsqrt_22 = None
        float_50 = (
            l_self_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_69 = 1.0 + float_50
        float_50 = None
        output_45 = output_44 * add_69
        output_44 = add_69 = None
        hidden_states_45 = output_45.type_as(hidden_states_44)
        output_45 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_5 = torch._C._nn.gelu(linear_39, approximate="tanh")
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_45 = l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_46 = gelu_5 * linear_40
        gelu_5 = linear_40 = None
        hidden_states_47 = torch.nn.functional.dropout(
            hidden_states_46, 0.0, False, False
        )
        hidden_states_46 = None
        down_proj_5 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_51 = down_proj_5.float()
        pow_24 = float_51.pow(2)
        mean_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_70 = mean_23 + 1e-06
        mean_23 = None
        rsqrt_23 = torch.rsqrt(add_70)
        add_70 = None
        output_46 = float_51 * rsqrt_23
        float_51 = rsqrt_23 = None
        float_52 = (
            l_self_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_71 = 1.0 + float_52
        float_52 = None
        output_47 = output_46 * add_71
        output_46 = add_71 = None
        hidden_states_48 = output_47.type_as(down_proj_5)
        output_47 = down_proj_5 = None
        dropout_18 = torch.nn.functional.dropout(hidden_states_48, 0.0, False, False)
        hidden_states_48 = None
        hidden_states_49 = hidden_states_44 + dropout_18
        hidden_states_44 = dropout_18 = None
        float_53 = hidden_states_49.float()
        pow_25 = float_53.pow(2)
        mean_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_73 = mean_24 + 1e-06
        mean_24 = None
        rsqrt_24 = torch.rsqrt(add_73)
        add_73 = None
        output_48 = float_53 * rsqrt_24
        float_53 = rsqrt_24 = None
        float_54 = (
            l_self_modules_layers_modules_6_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_74 = 1.0 + float_54
        float_54 = None
        output_49 = output_48 * add_74
        output_48 = add_74 = None
        hidden_states_50 = output_49.type_as(hidden_states_49)
        output_49 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 10, -1, 64))
        linear_42 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 10, -1, 64))
        linear_43 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_50 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_44.view((1, 10, -1, 64))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_83 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 32, None))]
        x2_12 = query_states_6[(Ellipsis, slice(32, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_84 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_83 + mul_84
        mul_83 = mul_84 = None
        mul_85 = key_states_6 * cos_9
        cos_9 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 32, None))]
        x2_13 = key_states_6[(Ellipsis, slice(32, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_86 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_85 + mul_86
        mul_85 = mul_86 = None
        attention_mask_8 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_6 = q_embed_6.contiguous()
        q_embed_6 = None
        key_6 = k_embed_6.contiguous()
        k_embed_6 = None
        value_6 = value_states_6.contiguous()
        value_states_6 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_8 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_6 = attn_output_25.reshape(1, 10, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_55 = attn_output_27.float()
        pow_26 = float_55.pow(2)
        mean_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_77 = mean_25 + 1e-06
        mean_25 = None
        rsqrt_25 = torch.rsqrt(add_77)
        add_77 = None
        output_50 = float_55 * rsqrt_25
        float_55 = rsqrt_25 = None
        float_56 = (
            l_self_modules_layers_modules_6_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_78 = 1.0 + float_56
        float_56 = None
        output_51 = output_50 * add_78
        output_50 = add_78 = None
        hidden_states_51 = output_51.type_as(attn_output_27)
        output_51 = attn_output_27 = None
        dropout_19 = torch.nn.functional.dropout(hidden_states_51, 0.0, False, False)
        hidden_states_51 = None
        hidden_states_52 = hidden_states_49 + dropout_19
        hidden_states_49 = dropout_19 = None
        float_57 = hidden_states_52.float()
        pow_27 = float_57.pow(2)
        mean_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_80 = mean_26 + 1e-06
        mean_26 = None
        rsqrt_26 = torch.rsqrt(add_80)
        add_80 = None
        output_52 = float_57 * rsqrt_26
        float_57 = rsqrt_26 = None
        float_58 = (
            l_self_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_81 = 1.0 + float_58
        float_58 = None
        output_53 = output_52 * add_81
        output_52 = add_81 = None
        hidden_states_53 = output_53.type_as(hidden_states_52)
        output_53 = None
        linear_46 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_6 = torch._C._nn.gelu(linear_46, approximate="tanh")
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_53 = l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_54 = gelu_6 * linear_47
        gelu_6 = linear_47 = None
        hidden_states_55 = torch.nn.functional.dropout(
            hidden_states_54, 0.0, False, False
        )
        hidden_states_54 = None
        down_proj_6 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_55 = l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_59 = down_proj_6.float()
        pow_28 = float_59.pow(2)
        mean_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_82 = mean_27 + 1e-06
        mean_27 = None
        rsqrt_27 = torch.rsqrt(add_82)
        add_82 = None
        output_54 = float_59 * rsqrt_27
        float_59 = rsqrt_27 = None
        float_60 = (
            l_self_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_83 = 1.0 + float_60
        float_60 = None
        output_55 = output_54 * add_83
        output_54 = add_83 = None
        hidden_states_56 = output_55.type_as(down_proj_6)
        output_55 = down_proj_6 = None
        dropout_21 = torch.nn.functional.dropout(hidden_states_56, 0.0, False, False)
        hidden_states_56 = None
        hidden_states_57 = hidden_states_52 + dropout_21
        hidden_states_52 = dropout_21 = None
        float_61 = hidden_states_57.float()
        pow_29 = float_61.pow(2)
        mean_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_85 = mean_28 + 1e-06
        mean_28 = None
        rsqrt_28 = torch.rsqrt(add_85)
        add_85 = None
        output_56 = float_61 * rsqrt_28
        float_61 = rsqrt_28 = None
        float_62 = (
            l_self_modules_layers_modules_7_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_86 = 1.0 + float_62
        float_62 = None
        output_57 = output_56 * add_86
        output_56 = add_86 = None
        hidden_states_58 = output_57.type_as(hidden_states_57)
        output_57 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 10, -1, 64))
        linear_49 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 10, -1, 64))
        linear_50 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_58 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_51.view((1, 10, -1, 64))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_96 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 32, None))]
        x2_14 = query_states_7[(Ellipsis, slice(32, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_97 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_96 + mul_97
        mul_96 = mul_97 = None
        mul_98 = key_states_7 * cos_10
        cos_10 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 32, None))]
        x2_15 = key_states_7[(Ellipsis, slice(32, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_99 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_98 + mul_99
        mul_98 = mul_99 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_7 = q_embed_7.contiguous()
        q_embed_7 = None
        key_7 = k_embed_7.contiguous()
        k_embed_7 = None
        value_7 = value_states_7.contiguous()
        value_states_7 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_9 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_7 = attn_output_29.reshape(1, 10, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_63 = attn_output_31.float()
        pow_30 = float_63.pow(2)
        mean_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_89 = mean_29 + 1e-06
        mean_29 = None
        rsqrt_29 = torch.rsqrt(add_89)
        add_89 = None
        output_58 = float_63 * rsqrt_29
        float_63 = rsqrt_29 = None
        float_64 = (
            l_self_modules_layers_modules_7_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_90 = 1.0 + float_64
        float_64 = None
        output_59 = output_58 * add_90
        output_58 = add_90 = None
        hidden_states_59 = output_59.type_as(attn_output_31)
        output_59 = attn_output_31 = None
        dropout_22 = torch.nn.functional.dropout(hidden_states_59, 0.0, False, False)
        hidden_states_59 = None
        hidden_states_60 = hidden_states_57 + dropout_22
        hidden_states_57 = dropout_22 = None
        float_65 = hidden_states_60.float()
        pow_31 = float_65.pow(2)
        mean_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_92 = mean_30 + 1e-06
        mean_30 = None
        rsqrt_30 = torch.rsqrt(add_92)
        add_92 = None
        output_60 = float_65 * rsqrt_30
        float_65 = rsqrt_30 = None
        float_66 = (
            l_self_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_93 = 1.0 + float_66
        float_66 = None
        output_61 = output_60 * add_93
        output_60 = add_93 = None
        hidden_states_61 = output_61.type_as(hidden_states_60)
        output_61 = None
        linear_53 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_7 = torch._C._nn.gelu(linear_53, approximate="tanh")
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_61 = l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_62 = gelu_7 * linear_54
        gelu_7 = linear_54 = None
        hidden_states_63 = torch.nn.functional.dropout(
            hidden_states_62, 0.0, False, False
        )
        hidden_states_62 = None
        down_proj_7 = torch._C._nn.linear(
            hidden_states_63,
            l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_63 = l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_67 = down_proj_7.float()
        pow_32 = float_67.pow(2)
        mean_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_94 = mean_31 + 1e-06
        mean_31 = None
        rsqrt_31 = torch.rsqrt(add_94)
        add_94 = None
        output_62 = float_67 * rsqrt_31
        float_67 = rsqrt_31 = None
        float_68 = (
            l_self_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_95 = 1.0 + float_68
        float_68 = None
        output_63 = output_62 * add_95
        output_62 = add_95 = None
        hidden_states_64 = output_63.type_as(down_proj_7)
        output_63 = down_proj_7 = None
        dropout_24 = torch.nn.functional.dropout(hidden_states_64, 0.0, False, False)
        hidden_states_64 = None
        hidden_states_65 = hidden_states_60 + dropout_24
        hidden_states_60 = dropout_24 = None
        float_69 = hidden_states_65.float()
        pow_33 = float_69.pow(2)
        mean_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_97 = mean_32 + 1e-06
        mean_32 = None
        rsqrt_32 = torch.rsqrt(add_97)
        add_97 = None
        output_64 = float_69 * rsqrt_32
        float_69 = rsqrt_32 = None
        float_70 = (
            l_self_modules_layers_modules_8_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_98 = 1.0 + float_70
        float_70 = None
        output_65 = output_64 * add_98
        output_64 = add_98 = None
        hidden_states_66 = output_65.type_as(hidden_states_65)
        output_65 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_56.view((1, 10, -1, 64))
        linear_56 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_57.view((1, 10, -1, 64))
        linear_57 = None
        key_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_66,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_66 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = linear_58.view((1, 10, -1, 64))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_109 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 32, None))]
        x2_16 = query_states_8[(Ellipsis, slice(32, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_110 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_109 + mul_110
        mul_109 = mul_110 = None
        mul_111 = key_states_8 * cos_11
        cos_11 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 32, None))]
        x2_17 = key_states_8[(Ellipsis, slice(32, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_112 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_111 + mul_112
        mul_111 = mul_112 = None
        attention_mask_10 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_8 = q_embed_8.contiguous()
        q_embed_8 = None
        key_8 = k_embed_8.contiguous()
        k_embed_8 = None
        value_8 = value_states_8.contiguous()
        value_states_8 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_10 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_8 = attn_output_33.reshape(1, 10, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_71 = attn_output_35.float()
        pow_34 = float_71.pow(2)
        mean_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_101 = mean_33 + 1e-06
        mean_33 = None
        rsqrt_33 = torch.rsqrt(add_101)
        add_101 = None
        output_66 = float_71 * rsqrt_33
        float_71 = rsqrt_33 = None
        float_72 = (
            l_self_modules_layers_modules_8_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_102 = 1.0 + float_72
        float_72 = None
        output_67 = output_66 * add_102
        output_66 = add_102 = None
        hidden_states_67 = output_67.type_as(attn_output_35)
        output_67 = attn_output_35 = None
        dropout_25 = torch.nn.functional.dropout(hidden_states_67, 0.0, False, False)
        hidden_states_67 = None
        hidden_states_68 = hidden_states_65 + dropout_25
        hidden_states_65 = dropout_25 = None
        float_73 = hidden_states_68.float()
        pow_35 = float_73.pow(2)
        mean_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_104 = mean_34 + 1e-06
        mean_34 = None
        rsqrt_34 = torch.rsqrt(add_104)
        add_104 = None
        output_68 = float_73 * rsqrt_34
        float_73 = rsqrt_34 = None
        float_74 = (
            l_self_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_105 = 1.0 + float_74
        float_74 = None
        output_69 = output_68 * add_105
        output_68 = add_105 = None
        hidden_states_69 = output_69.type_as(hidden_states_68)
        output_69 = None
        linear_60 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_8 = torch._C._nn.gelu(linear_60, approximate="tanh")
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_69 = l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_70 = gelu_8 * linear_61
        gelu_8 = linear_61 = None
        hidden_states_71 = torch.nn.functional.dropout(
            hidden_states_70, 0.0, False, False
        )
        hidden_states_70 = None
        down_proj_8 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_71 = l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_75 = down_proj_8.float()
        pow_36 = float_75.pow(2)
        mean_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_106 = mean_35 + 1e-06
        mean_35 = None
        rsqrt_35 = torch.rsqrt(add_106)
        add_106 = None
        output_70 = float_75 * rsqrt_35
        float_75 = rsqrt_35 = None
        float_76 = (
            l_self_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_107 = 1.0 + float_76
        float_76 = None
        output_71 = output_70 * add_107
        output_70 = add_107 = None
        hidden_states_72 = output_71.type_as(down_proj_8)
        output_71 = down_proj_8 = None
        dropout_27 = torch.nn.functional.dropout(hidden_states_72, 0.0, False, False)
        hidden_states_72 = None
        hidden_states_73 = hidden_states_68 + dropout_27
        hidden_states_68 = dropout_27 = None
        float_77 = hidden_states_73.float()
        pow_37 = float_77.pow(2)
        mean_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_109 = mean_36 + 1e-06
        mean_36 = None
        rsqrt_36 = torch.rsqrt(add_109)
        add_109 = None
        output_72 = float_77 * rsqrt_36
        float_77 = rsqrt_36 = None
        float_78 = (
            l_self_modules_layers_modules_9_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_110 = 1.0 + float_78
        float_78 = None
        output_73 = output_72 * add_110
        output_72 = add_110 = None
        hidden_states_74 = output_73.type_as(hidden_states_73)
        output_73 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_63.view((1, 10, -1, 64))
        linear_63 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_64.view((1, 10, -1, 64))
        linear_64 = None
        key_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_74 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = linear_65.view((1, 10, -1, 64))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_122 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 32, None))]
        x2_18 = query_states_9[(Ellipsis, slice(32, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_123 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_122 + mul_123
        mul_122 = mul_123 = None
        mul_124 = key_states_9 * cos_12
        cos_12 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 32, None))]
        x2_19 = key_states_9[(Ellipsis, slice(32, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_125 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_124 + mul_125
        mul_124 = mul_125 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_9 = q_embed_9.contiguous()
        q_embed_9 = None
        key_9 = k_embed_9.contiguous()
        k_embed_9 = None
        value_9 = value_states_9.contiguous()
        value_states_9 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_11 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_9 = attn_output_37.reshape(1, 10, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_79 = attn_output_39.float()
        pow_38 = float_79.pow(2)
        mean_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_113 = mean_37 + 1e-06
        mean_37 = None
        rsqrt_37 = torch.rsqrt(add_113)
        add_113 = None
        output_74 = float_79 * rsqrt_37
        float_79 = rsqrt_37 = None
        float_80 = (
            l_self_modules_layers_modules_9_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_114 = 1.0 + float_80
        float_80 = None
        output_75 = output_74 * add_114
        output_74 = add_114 = None
        hidden_states_75 = output_75.type_as(attn_output_39)
        output_75 = attn_output_39 = None
        dropout_28 = torch.nn.functional.dropout(hidden_states_75, 0.0, False, False)
        hidden_states_75 = None
        hidden_states_76 = hidden_states_73 + dropout_28
        hidden_states_73 = dropout_28 = None
        float_81 = hidden_states_76.float()
        pow_39 = float_81.pow(2)
        mean_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_116 = mean_38 + 1e-06
        mean_38 = None
        rsqrt_38 = torch.rsqrt(add_116)
        add_116 = None
        output_76 = float_81 * rsqrt_38
        float_81 = rsqrt_38 = None
        float_82 = (
            l_self_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_117 = 1.0 + float_82
        float_82 = None
        output_77 = output_76 * add_117
        output_76 = add_117 = None
        hidden_states_77 = output_77.type_as(hidden_states_76)
        output_77 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_9 = torch._C._nn.gelu(linear_67, approximate="tanh")
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_77 = l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_78 = gelu_9 * linear_68
        gelu_9 = linear_68 = None
        hidden_states_79 = torch.nn.functional.dropout(
            hidden_states_78, 0.0, False, False
        )
        hidden_states_78 = None
        down_proj_9 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_79 = l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_83 = down_proj_9.float()
        pow_40 = float_83.pow(2)
        mean_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_118 = mean_39 + 1e-06
        mean_39 = None
        rsqrt_39 = torch.rsqrt(add_118)
        add_118 = None
        output_78 = float_83 * rsqrt_39
        float_83 = rsqrt_39 = None
        float_84 = (
            l_self_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_119 = 1.0 + float_84
        float_84 = None
        output_79 = output_78 * add_119
        output_78 = add_119 = None
        hidden_states_80 = output_79.type_as(down_proj_9)
        output_79 = down_proj_9 = None
        dropout_30 = torch.nn.functional.dropout(hidden_states_80, 0.0, False, False)
        hidden_states_80 = None
        hidden_states_81 = hidden_states_76 + dropout_30
        hidden_states_76 = dropout_30 = None
        float_85 = hidden_states_81.float()
        pow_41 = float_85.pow(2)
        mean_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_121 = mean_40 + 1e-06
        mean_40 = None
        rsqrt_40 = torch.rsqrt(add_121)
        add_121 = None
        output_80 = float_85 * rsqrt_40
        float_85 = rsqrt_40 = None
        float_86 = (
            l_self_modules_layers_modules_10_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_122 = 1.0 + float_86
        float_86 = None
        output_81 = output_80 * add_122
        output_80 = add_122 = None
        hidden_states_82 = output_81.type_as(hidden_states_81)
        output_81 = None
        linear_70 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_70.view((1, 10, -1, 64))
        linear_70 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_71.view((1, 10, -1, 64))
        linear_71 = None
        key_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_82 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = linear_72.view((1, 10, -1, 64))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_135 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 32, None))]
        x2_20 = query_states_10[(Ellipsis, slice(32, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_136 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_135 + mul_136
        mul_135 = mul_136 = None
        mul_137 = key_states_10 * cos_13
        cos_13 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 32, None))]
        x2_21 = key_states_10[(Ellipsis, slice(32, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_138 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_137 + mul_138
        mul_137 = mul_138 = None
        attention_mask_12 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_10 = q_embed_10.contiguous()
        q_embed_10 = None
        key_10 = k_embed_10.contiguous()
        k_embed_10 = None
        value_10 = value_states_10.contiguous()
        value_states_10 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_12 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_10 = attn_output_41.reshape(1, 10, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_87 = attn_output_43.float()
        pow_42 = float_87.pow(2)
        mean_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_125 = mean_41 + 1e-06
        mean_41 = None
        rsqrt_41 = torch.rsqrt(add_125)
        add_125 = None
        output_82 = float_87 * rsqrt_41
        float_87 = rsqrt_41 = None
        float_88 = (
            l_self_modules_layers_modules_10_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_126 = 1.0 + float_88
        float_88 = None
        output_83 = output_82 * add_126
        output_82 = add_126 = None
        hidden_states_83 = output_83.type_as(attn_output_43)
        output_83 = attn_output_43 = None
        dropout_31 = torch.nn.functional.dropout(hidden_states_83, 0.0, False, False)
        hidden_states_83 = None
        hidden_states_84 = hidden_states_81 + dropout_31
        hidden_states_81 = dropout_31 = None
        float_89 = hidden_states_84.float()
        pow_43 = float_89.pow(2)
        mean_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_128 = mean_42 + 1e-06
        mean_42 = None
        rsqrt_42 = torch.rsqrt(add_128)
        add_128 = None
        output_84 = float_89 * rsqrt_42
        float_89 = rsqrt_42 = None
        float_90 = (
            l_self_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_129 = 1.0 + float_90
        float_90 = None
        output_85 = output_84 * add_129
        output_84 = add_129 = None
        hidden_states_85 = output_85.type_as(hidden_states_84)
        output_85 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_10 = torch._C._nn.gelu(linear_74, approximate="tanh")
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_85 = l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_86 = gelu_10 * linear_75
        gelu_10 = linear_75 = None
        hidden_states_87 = torch.nn.functional.dropout(
            hidden_states_86, 0.0, False, False
        )
        hidden_states_86 = None
        down_proj_10 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_87 = l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_91 = down_proj_10.float()
        pow_44 = float_91.pow(2)
        mean_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_130 = mean_43 + 1e-06
        mean_43 = None
        rsqrt_43 = torch.rsqrt(add_130)
        add_130 = None
        output_86 = float_91 * rsqrt_43
        float_91 = rsqrt_43 = None
        float_92 = (
            l_self_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_131 = 1.0 + float_92
        float_92 = None
        output_87 = output_86 * add_131
        output_86 = add_131 = None
        hidden_states_88 = output_87.type_as(down_proj_10)
        output_87 = down_proj_10 = None
        dropout_33 = torch.nn.functional.dropout(hidden_states_88, 0.0, False, False)
        hidden_states_88 = None
        hidden_states_89 = hidden_states_84 + dropout_33
        hidden_states_84 = dropout_33 = None
        float_93 = hidden_states_89.float()
        pow_45 = float_93.pow(2)
        mean_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_133 = mean_44 + 1e-06
        mean_44 = None
        rsqrt_44 = torch.rsqrt(add_133)
        add_133 = None
        output_88 = float_93 * rsqrt_44
        float_93 = rsqrt_44 = None
        float_94 = (
            l_self_modules_layers_modules_11_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_134 = 1.0 + float_94
        float_94 = None
        output_89 = output_88 * add_134
        output_88 = add_134 = None
        hidden_states_90 = output_89.type_as(hidden_states_89)
        output_89 = None
        linear_77 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_77.view((1, 10, -1, 64))
        linear_77 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_78.view((1, 10, -1, 64))
        linear_78 = None
        key_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_90 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = linear_79.view((1, 10, -1, 64))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_148 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 32, None))]
        x2_22 = query_states_11[(Ellipsis, slice(32, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_149 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_148 + mul_149
        mul_148 = mul_149 = None
        mul_150 = key_states_11 * cos_14
        cos_14 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 32, None))]
        x2_23 = key_states_11[(Ellipsis, slice(32, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_151 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_150 + mul_151
        mul_150 = mul_151 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_11 = q_embed_11.contiguous()
        q_embed_11 = None
        key_11 = k_embed_11.contiguous()
        k_embed_11 = None
        value_11 = value_states_11.contiguous()
        value_states_11 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_13 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_11 = attn_output_45.reshape(1, 10, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_95 = attn_output_47.float()
        pow_46 = float_95.pow(2)
        mean_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_137 = mean_45 + 1e-06
        mean_45 = None
        rsqrt_45 = torch.rsqrt(add_137)
        add_137 = None
        output_90 = float_95 * rsqrt_45
        float_95 = rsqrt_45 = None
        float_96 = (
            l_self_modules_layers_modules_11_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_138 = 1.0 + float_96
        float_96 = None
        output_91 = output_90 * add_138
        output_90 = add_138 = None
        hidden_states_91 = output_91.type_as(attn_output_47)
        output_91 = attn_output_47 = None
        dropout_34 = torch.nn.functional.dropout(hidden_states_91, 0.0, False, False)
        hidden_states_91 = None
        hidden_states_92 = hidden_states_89 + dropout_34
        hidden_states_89 = dropout_34 = None
        float_97 = hidden_states_92.float()
        pow_47 = float_97.pow(2)
        mean_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_140 = mean_46 + 1e-06
        mean_46 = None
        rsqrt_46 = torch.rsqrt(add_140)
        add_140 = None
        output_92 = float_97 * rsqrt_46
        float_97 = rsqrt_46 = None
        float_98 = (
            l_self_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_141 = 1.0 + float_98
        float_98 = None
        output_93 = output_92 * add_141
        output_92 = add_141 = None
        hidden_states_93 = output_93.type_as(hidden_states_92)
        output_93 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_11 = torch._C._nn.gelu(linear_81, approximate="tanh")
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_93 = l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_94 = gelu_11 * linear_82
        gelu_11 = linear_82 = None
        hidden_states_95 = torch.nn.functional.dropout(
            hidden_states_94, 0.0, False, False
        )
        hidden_states_94 = None
        down_proj_11 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_95 = l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_99 = down_proj_11.float()
        pow_48 = float_99.pow(2)
        mean_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_142 = mean_47 + 1e-06
        mean_47 = None
        rsqrt_47 = torch.rsqrt(add_142)
        add_142 = None
        output_94 = float_99 * rsqrt_47
        float_99 = rsqrt_47 = None
        float_100 = (
            l_self_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_143 = 1.0 + float_100
        float_100 = None
        output_95 = output_94 * add_143
        output_94 = add_143 = None
        hidden_states_96 = output_95.type_as(down_proj_11)
        output_95 = down_proj_11 = None
        dropout_36 = torch.nn.functional.dropout(hidden_states_96, 0.0, False, False)
        hidden_states_96 = None
        hidden_states_97 = hidden_states_92 + dropout_36
        hidden_states_92 = dropout_36 = None
        float_101 = hidden_states_97.float()
        pow_49 = float_101.pow(2)
        mean_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_145 = mean_48 + 1e-06
        mean_48 = None
        rsqrt_48 = torch.rsqrt(add_145)
        add_145 = None
        output_96 = float_101 * rsqrt_48
        float_101 = rsqrt_48 = None
        float_102 = (
            l_self_modules_layers_modules_12_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_146 = 1.0 + float_102
        float_102 = None
        output_97 = output_96 * add_146
        output_96 = add_146 = None
        hidden_states_98 = output_97.type_as(hidden_states_97)
        output_97 = None
        linear_84 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_36 = linear_84.view((1, 10, -1, 64))
        linear_84 = None
        query_states_12 = view_36.transpose(1, 2)
        view_36 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_85.view((1, 10, -1, 64))
        linear_85 = None
        key_states_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_98 = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_38 = linear_86.view((1, 10, -1, 64))
        linear_86 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_161 = query_states_12 * cos_15
        x1_24 = query_states_12[(Ellipsis, slice(None, 32, None))]
        x2_24 = query_states_12[(Ellipsis, slice(32, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_162 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_161 + mul_162
        mul_161 = mul_162 = None
        mul_163 = key_states_12 * cos_15
        cos_15 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 32, None))]
        x2_25 = key_states_12[(Ellipsis, slice(32, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_164 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_163 + mul_164
        mul_163 = mul_164 = None
        attention_mask_14 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_12 = q_embed_12.contiguous()
        q_embed_12 = None
        key_12 = k_embed_12.contiguous()
        k_embed_12 = None
        value_12 = value_states_12.contiguous()
        value_states_12 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_14 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_12 = attn_output_49.reshape(1, 10, -1)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_103 = attn_output_51.float()
        pow_50 = float_103.pow(2)
        mean_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_149 = mean_49 + 1e-06
        mean_49 = None
        rsqrt_49 = torch.rsqrt(add_149)
        add_149 = None
        output_98 = float_103 * rsqrt_49
        float_103 = rsqrt_49 = None
        float_104 = (
            l_self_modules_layers_modules_12_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_150 = 1.0 + float_104
        float_104 = None
        output_99 = output_98 * add_150
        output_98 = add_150 = None
        hidden_states_99 = output_99.type_as(attn_output_51)
        output_99 = attn_output_51 = None
        dropout_37 = torch.nn.functional.dropout(hidden_states_99, 0.0, False, False)
        hidden_states_99 = None
        hidden_states_100 = hidden_states_97 + dropout_37
        hidden_states_97 = dropout_37 = None
        float_105 = hidden_states_100.float()
        pow_51 = float_105.pow(2)
        mean_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_152 = mean_50 + 1e-06
        mean_50 = None
        rsqrt_50 = torch.rsqrt(add_152)
        add_152 = None
        output_100 = float_105 * rsqrt_50
        float_105 = rsqrt_50 = None
        float_106 = (
            l_self_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_153 = 1.0 + float_106
        float_106 = None
        output_101 = output_100 * add_153
        output_100 = add_153 = None
        hidden_states_101 = output_101.type_as(hidden_states_100)
        output_101 = None
        linear_88 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_12 = torch._C._nn.gelu(linear_88, approximate="tanh")
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_101 = l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_102 = gelu_12 * linear_89
        gelu_12 = linear_89 = None
        hidden_states_103 = torch.nn.functional.dropout(
            hidden_states_102, 0.0, False, False
        )
        hidden_states_102 = None
        down_proj_12 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_103 = l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_107 = down_proj_12.float()
        pow_52 = float_107.pow(2)
        mean_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_154 = mean_51 + 1e-06
        mean_51 = None
        rsqrt_51 = torch.rsqrt(add_154)
        add_154 = None
        output_102 = float_107 * rsqrt_51
        float_107 = rsqrt_51 = None
        float_108 = (
            l_self_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_155 = 1.0 + float_108
        float_108 = None
        output_103 = output_102 * add_155
        output_102 = add_155 = None
        hidden_states_104 = output_103.type_as(down_proj_12)
        output_103 = down_proj_12 = None
        dropout_39 = torch.nn.functional.dropout(hidden_states_104, 0.0, False, False)
        hidden_states_104 = None
        hidden_states_105 = hidden_states_100 + dropout_39
        hidden_states_100 = dropout_39 = None
        float_109 = hidden_states_105.float()
        pow_53 = float_109.pow(2)
        mean_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_157 = mean_52 + 1e-06
        mean_52 = None
        rsqrt_52 = torch.rsqrt(add_157)
        add_157 = None
        output_104 = float_109 * rsqrt_52
        float_109 = rsqrt_52 = None
        float_110 = (
            l_self_modules_layers_modules_13_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_158 = 1.0 + float_110
        float_110 = None
        output_105 = output_104 * add_158
        output_104 = add_158 = None
        hidden_states_106 = output_105.type_as(hidden_states_105)
        output_105 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_39 = linear_91.view((1, 10, -1, 64))
        linear_91 = None
        query_states_13 = view_39.transpose(1, 2)
        view_39 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_92.view((1, 10, -1, 64))
        linear_92 = None
        key_states_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_106,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_106 = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_41 = linear_93.view((1, 10, -1, 64))
        linear_93 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_174 = query_states_13 * cos_16
        x1_26 = query_states_13[(Ellipsis, slice(None, 32, None))]
        x2_26 = query_states_13[(Ellipsis, slice(32, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_175 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_174 + mul_175
        mul_174 = mul_175 = None
        mul_176 = key_states_13 * cos_16
        cos_16 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 32, None))]
        x2_27 = key_states_13[(Ellipsis, slice(32, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_177 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_176 + mul_177
        mul_176 = mul_177 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_13 = q_embed_13.contiguous()
        q_embed_13 = None
        key_13 = k_embed_13.contiguous()
        k_embed_13 = None
        value_13 = value_states_13.contiguous()
        value_states_13 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_15 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_13 = attn_output_53.reshape(1, 10, -1)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_111 = attn_output_55.float()
        pow_54 = float_111.pow(2)
        mean_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_161 = mean_53 + 1e-06
        mean_53 = None
        rsqrt_53 = torch.rsqrt(add_161)
        add_161 = None
        output_106 = float_111 * rsqrt_53
        float_111 = rsqrt_53 = None
        float_112 = (
            l_self_modules_layers_modules_13_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_162 = 1.0 + float_112
        float_112 = None
        output_107 = output_106 * add_162
        output_106 = add_162 = None
        hidden_states_107 = output_107.type_as(attn_output_55)
        output_107 = attn_output_55 = None
        dropout_40 = torch.nn.functional.dropout(hidden_states_107, 0.0, False, False)
        hidden_states_107 = None
        hidden_states_108 = hidden_states_105 + dropout_40
        hidden_states_105 = dropout_40 = None
        float_113 = hidden_states_108.float()
        pow_55 = float_113.pow(2)
        mean_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_164 = mean_54 + 1e-06
        mean_54 = None
        rsqrt_54 = torch.rsqrt(add_164)
        add_164 = None
        output_108 = float_113 * rsqrt_54
        float_113 = rsqrt_54 = None
        float_114 = (
            l_self_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_165 = 1.0 + float_114
        float_114 = None
        output_109 = output_108 * add_165
        output_108 = add_165 = None
        hidden_states_109 = output_109.type_as(hidden_states_108)
        output_109 = None
        linear_95 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_13 = torch._C._nn.gelu(linear_95, approximate="tanh")
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_109 = l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_110 = gelu_13 * linear_96
        gelu_13 = linear_96 = None
        hidden_states_111 = torch.nn.functional.dropout(
            hidden_states_110, 0.0, False, False
        )
        hidden_states_110 = None
        down_proj_13 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_111 = l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_115 = down_proj_13.float()
        pow_56 = float_115.pow(2)
        mean_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_166 = mean_55 + 1e-06
        mean_55 = None
        rsqrt_55 = torch.rsqrt(add_166)
        add_166 = None
        output_110 = float_115 * rsqrt_55
        float_115 = rsqrt_55 = None
        float_116 = (
            l_self_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_167 = 1.0 + float_116
        float_116 = None
        output_111 = output_110 * add_167
        output_110 = add_167 = None
        hidden_states_112 = output_111.type_as(down_proj_13)
        output_111 = down_proj_13 = None
        dropout_42 = torch.nn.functional.dropout(hidden_states_112, 0.0, False, False)
        hidden_states_112 = None
        hidden_states_113 = hidden_states_108 + dropout_42
        hidden_states_108 = dropout_42 = None
        float_117 = hidden_states_113.float()
        pow_57 = float_117.pow(2)
        mean_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_169 = mean_56 + 1e-06
        mean_56 = None
        rsqrt_56 = torch.rsqrt(add_169)
        add_169 = None
        output_112 = float_117 * rsqrt_56
        float_117 = rsqrt_56 = None
        float_118 = (
            l_self_modules_layers_modules_14_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_170 = 1.0 + float_118
        float_118 = None
        output_113 = output_112 * add_170
        output_112 = add_170 = None
        hidden_states_114 = output_113.type_as(hidden_states_113)
        output_113 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_42 = linear_98.view((1, 10, -1, 64))
        linear_98 = None
        query_states_14 = view_42.transpose(1, 2)
        view_42 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_99.view((1, 10, -1, 64))
        linear_99 = None
        key_states_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_114 = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_44 = linear_100.view((1, 10, -1, 64))
        linear_100 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_187 = query_states_14 * cos_17
        x1_28 = query_states_14[(Ellipsis, slice(None, 32, None))]
        x2_28 = query_states_14[(Ellipsis, slice(32, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_188 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_187 + mul_188
        mul_187 = mul_188 = None
        mul_189 = key_states_14 * cos_17
        cos_17 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 32, None))]
        x2_29 = key_states_14[(Ellipsis, slice(32, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_190 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_189 + mul_190
        mul_189 = mul_190 = None
        attention_mask_16 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_14 = q_embed_14.contiguous()
        q_embed_14 = None
        key_14 = k_embed_14.contiguous()
        k_embed_14 = None
        value_14 = value_states_14.contiguous()
        value_states_14 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_16 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_14 = attn_output_57.reshape(1, 10, -1)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_119 = attn_output_59.float()
        pow_58 = float_119.pow(2)
        mean_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_173 = mean_57 + 1e-06
        mean_57 = None
        rsqrt_57 = torch.rsqrt(add_173)
        add_173 = None
        output_114 = float_119 * rsqrt_57
        float_119 = rsqrt_57 = None
        float_120 = (
            l_self_modules_layers_modules_14_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_174 = 1.0 + float_120
        float_120 = None
        output_115 = output_114 * add_174
        output_114 = add_174 = None
        hidden_states_115 = output_115.type_as(attn_output_59)
        output_115 = attn_output_59 = None
        dropout_43 = torch.nn.functional.dropout(hidden_states_115, 0.0, False, False)
        hidden_states_115 = None
        hidden_states_116 = hidden_states_113 + dropout_43
        hidden_states_113 = dropout_43 = None
        float_121 = hidden_states_116.float()
        pow_59 = float_121.pow(2)
        mean_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_176 = mean_58 + 1e-06
        mean_58 = None
        rsqrt_58 = torch.rsqrt(add_176)
        add_176 = None
        output_116 = float_121 * rsqrt_58
        float_121 = rsqrt_58 = None
        float_122 = (
            l_self_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_177 = 1.0 + float_122
        float_122 = None
        output_117 = output_116 * add_177
        output_116 = add_177 = None
        hidden_states_117 = output_117.type_as(hidden_states_116)
        output_117 = None
        linear_102 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_14 = torch._C._nn.gelu(linear_102, approximate="tanh")
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_117 = l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_118 = gelu_14 * linear_103
        gelu_14 = linear_103 = None
        hidden_states_119 = torch.nn.functional.dropout(
            hidden_states_118, 0.0, False, False
        )
        hidden_states_118 = None
        down_proj_14 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_119 = l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_123 = down_proj_14.float()
        pow_60 = float_123.pow(2)
        mean_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_178 = mean_59 + 1e-06
        mean_59 = None
        rsqrt_59 = torch.rsqrt(add_178)
        add_178 = None
        output_118 = float_123 * rsqrt_59
        float_123 = rsqrt_59 = None
        float_124 = (
            l_self_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_179 = 1.0 + float_124
        float_124 = None
        output_119 = output_118 * add_179
        output_118 = add_179 = None
        hidden_states_120 = output_119.type_as(down_proj_14)
        output_119 = down_proj_14 = None
        dropout_45 = torch.nn.functional.dropout(hidden_states_120, 0.0, False, False)
        hidden_states_120 = None
        hidden_states_121 = hidden_states_116 + dropout_45
        hidden_states_116 = dropout_45 = None
        float_125 = hidden_states_121.float()
        pow_61 = float_125.pow(2)
        mean_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_181 = mean_60 + 1e-06
        mean_60 = None
        rsqrt_60 = torch.rsqrt(add_181)
        add_181 = None
        output_120 = float_125 * rsqrt_60
        float_125 = rsqrt_60 = None
        float_126 = (
            l_self_modules_layers_modules_15_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_182 = 1.0 + float_126
        float_126 = None
        output_121 = output_120 * add_182
        output_120 = add_182 = None
        hidden_states_122 = output_121.type_as(hidden_states_121)
        output_121 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_45 = linear_105.view((1, 10, -1, 64))
        linear_105 = None
        query_states_15 = view_45.transpose(1, 2)
        view_45 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_106.view((1, 10, -1, 64))
        linear_106 = None
        key_states_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_122 = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_47 = linear_107.view((1, 10, -1, 64))
        linear_107 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_200 = query_states_15 * cos_18
        x1_30 = query_states_15[(Ellipsis, slice(None, 32, None))]
        x2_30 = query_states_15[(Ellipsis, slice(32, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_201 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_200 + mul_201
        mul_200 = mul_201 = None
        mul_202 = key_states_15 * cos_18
        cos_18 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 32, None))]
        x2_31 = key_states_15[(Ellipsis, slice(32, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_203 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_202 + mul_203
        mul_202 = mul_203 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_15 = q_embed_15.contiguous()
        q_embed_15 = None
        key_15 = k_embed_15.contiguous()
        k_embed_15 = None
        value_15 = value_states_15.contiguous()
        value_states_15 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_17 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_15 = attn_output_61.reshape(1, 10, -1)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_127 = attn_output_63.float()
        pow_62 = float_127.pow(2)
        mean_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_185 = mean_61 + 1e-06
        mean_61 = None
        rsqrt_61 = torch.rsqrt(add_185)
        add_185 = None
        output_122 = float_127 * rsqrt_61
        float_127 = rsqrt_61 = None
        float_128 = (
            l_self_modules_layers_modules_15_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_186 = 1.0 + float_128
        float_128 = None
        output_123 = output_122 * add_186
        output_122 = add_186 = None
        hidden_states_123 = output_123.type_as(attn_output_63)
        output_123 = attn_output_63 = None
        dropout_46 = torch.nn.functional.dropout(hidden_states_123, 0.0, False, False)
        hidden_states_123 = None
        hidden_states_124 = hidden_states_121 + dropout_46
        hidden_states_121 = dropout_46 = None
        float_129 = hidden_states_124.float()
        pow_63 = float_129.pow(2)
        mean_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_188 = mean_62 + 1e-06
        mean_62 = None
        rsqrt_62 = torch.rsqrt(add_188)
        add_188 = None
        output_124 = float_129 * rsqrt_62
        float_129 = rsqrt_62 = None
        float_130 = (
            l_self_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_189 = 1.0 + float_130
        float_130 = None
        output_125 = output_124 * add_189
        output_124 = add_189 = None
        hidden_states_125 = output_125.type_as(hidden_states_124)
        output_125 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_15 = torch._C._nn.gelu(linear_109, approximate="tanh")
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_125 = l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_126 = gelu_15 * linear_110
        gelu_15 = linear_110 = None
        hidden_states_127 = torch.nn.functional.dropout(
            hidden_states_126, 0.0, False, False
        )
        hidden_states_126 = None
        down_proj_15 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_127 = l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_131 = down_proj_15.float()
        pow_64 = float_131.pow(2)
        mean_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_190 = mean_63 + 1e-06
        mean_63 = None
        rsqrt_63 = torch.rsqrt(add_190)
        add_190 = None
        output_126 = float_131 * rsqrt_63
        float_131 = rsqrt_63 = None
        float_132 = (
            l_self_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_191 = 1.0 + float_132
        float_132 = None
        output_127 = output_126 * add_191
        output_126 = add_191 = None
        hidden_states_128 = output_127.type_as(down_proj_15)
        output_127 = down_proj_15 = None
        dropout_48 = torch.nn.functional.dropout(hidden_states_128, 0.0, False, False)
        hidden_states_128 = None
        hidden_states_129 = hidden_states_124 + dropout_48
        hidden_states_124 = dropout_48 = None
        float_133 = hidden_states_129.float()
        pow_65 = float_133.pow(2)
        mean_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_193 = mean_64 + 1e-06
        mean_64 = None
        rsqrt_64 = torch.rsqrt(add_193)
        add_193 = None
        output_128 = float_133 * rsqrt_64
        float_133 = rsqrt_64 = None
        float_134 = (
            l_self_modules_layers_modules_16_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_194 = 1.0 + float_134
        float_134 = None
        output_129 = output_128 * add_194
        output_128 = add_194 = None
        hidden_states_130 = output_129.type_as(hidden_states_129)
        output_129 = None
        linear_112 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_48 = linear_112.view((1, 10, -1, 64))
        linear_112 = None
        query_states_16 = view_48.transpose(1, 2)
        view_48 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_113.view((1, 10, -1, 64))
        linear_113 = None
        key_states_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_130 = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_50 = linear_114.view((1, 10, -1, 64))
        linear_114 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_213 = query_states_16 * cos_19
        x1_32 = query_states_16[(Ellipsis, slice(None, 32, None))]
        x2_32 = query_states_16[(Ellipsis, slice(32, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_214 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_213 + mul_214
        mul_213 = mul_214 = None
        mul_215 = key_states_16 * cos_19
        cos_19 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 32, None))]
        x2_33 = key_states_16[(Ellipsis, slice(32, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_216 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_215 + mul_216
        mul_215 = mul_216 = None
        attention_mask_18 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_16 = q_embed_16.contiguous()
        q_embed_16 = None
        key_16 = k_embed_16.contiguous()
        k_embed_16 = None
        value_16 = value_states_16.contiguous()
        value_states_16 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_18 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_16 = attn_output_65.reshape(1, 10, -1)
        attn_output_65 = None
        attn_output_66 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_135 = attn_output_67.float()
        pow_66 = float_135.pow(2)
        mean_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_197 = mean_65 + 1e-06
        mean_65 = None
        rsqrt_65 = torch.rsqrt(add_197)
        add_197 = None
        output_130 = float_135 * rsqrt_65
        float_135 = rsqrt_65 = None
        float_136 = (
            l_self_modules_layers_modules_16_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_198 = 1.0 + float_136
        float_136 = None
        output_131 = output_130 * add_198
        output_130 = add_198 = None
        hidden_states_131 = output_131.type_as(attn_output_67)
        output_131 = attn_output_67 = None
        dropout_49 = torch.nn.functional.dropout(hidden_states_131, 0.0, False, False)
        hidden_states_131 = None
        hidden_states_132 = hidden_states_129 + dropout_49
        hidden_states_129 = dropout_49 = None
        float_137 = hidden_states_132.float()
        pow_67 = float_137.pow(2)
        mean_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_200 = mean_66 + 1e-06
        mean_66 = None
        rsqrt_66 = torch.rsqrt(add_200)
        add_200 = None
        output_132 = float_137 * rsqrt_66
        float_137 = rsqrt_66 = None
        float_138 = (
            l_self_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_201 = 1.0 + float_138
        float_138 = None
        output_133 = output_132 * add_201
        output_132 = add_201 = None
        hidden_states_133 = output_133.type_as(hidden_states_132)
        output_133 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_16 = torch._C._nn.gelu(linear_116, approximate="tanh")
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_133 = l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_134 = gelu_16 * linear_117
        gelu_16 = linear_117 = None
        hidden_states_135 = torch.nn.functional.dropout(
            hidden_states_134, 0.0, False, False
        )
        hidden_states_134 = None
        down_proj_16 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_135 = l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_139 = down_proj_16.float()
        pow_68 = float_139.pow(2)
        mean_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_202 = mean_67 + 1e-06
        mean_67 = None
        rsqrt_67 = torch.rsqrt(add_202)
        add_202 = None
        output_134 = float_139 * rsqrt_67
        float_139 = rsqrt_67 = None
        float_140 = (
            l_self_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_203 = 1.0 + float_140
        float_140 = None
        output_135 = output_134 * add_203
        output_134 = add_203 = None
        hidden_states_136 = output_135.type_as(down_proj_16)
        output_135 = down_proj_16 = None
        dropout_51 = torch.nn.functional.dropout(hidden_states_136, 0.0, False, False)
        hidden_states_136 = None
        hidden_states_137 = hidden_states_132 + dropout_51
        hidden_states_132 = dropout_51 = None
        float_141 = hidden_states_137.float()
        pow_69 = float_141.pow(2)
        mean_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_205 = mean_68 + 1e-06
        mean_68 = None
        rsqrt_68 = torch.rsqrt(add_205)
        add_205 = None
        output_136 = float_141 * rsqrt_68
        float_141 = rsqrt_68 = None
        float_142 = (
            l_self_modules_layers_modules_17_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_206 = 1.0 + float_142
        float_142 = None
        output_137 = output_136 * add_206
        output_136 = add_206 = None
        hidden_states_138 = output_137.type_as(hidden_states_137)
        output_137 = None
        linear_119 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_51 = linear_119.view((1, 10, -1, 64))
        linear_119 = None
        query_states_17 = view_51.transpose(1, 2)
        view_51 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_120.view((1, 10, -1, 64))
        linear_120 = None
        key_states_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_138 = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_53 = linear_121.view((1, 10, -1, 64))
        linear_121 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_226 = query_states_17 * cos_20
        x1_34 = query_states_17[(Ellipsis, slice(None, 32, None))]
        x2_34 = query_states_17[(Ellipsis, slice(32, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_227 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_226 + mul_227
        mul_226 = mul_227 = None
        mul_228 = key_states_17 * cos_20
        cos_20 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 32, None))]
        x2_35 = key_states_17[(Ellipsis, slice(32, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_229 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_228 + mul_229
        mul_228 = mul_229 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_17 = q_embed_17.contiguous()
        q_embed_17 = None
        key_17 = k_embed_17.contiguous()
        k_embed_17 = None
        value_17 = value_states_17.contiguous()
        value_states_17 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_19 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_17 = attn_output_69.reshape(1, 10, -1)
        attn_output_69 = None
        attn_output_70 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_143 = attn_output_71.float()
        pow_70 = float_143.pow(2)
        mean_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_209 = mean_69 + 1e-06
        mean_69 = None
        rsqrt_69 = torch.rsqrt(add_209)
        add_209 = None
        output_138 = float_143 * rsqrt_69
        float_143 = rsqrt_69 = None
        float_144 = (
            l_self_modules_layers_modules_17_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_210 = 1.0 + float_144
        float_144 = None
        output_139 = output_138 * add_210
        output_138 = add_210 = None
        hidden_states_139 = output_139.type_as(attn_output_71)
        output_139 = attn_output_71 = None
        dropout_52 = torch.nn.functional.dropout(hidden_states_139, 0.0, False, False)
        hidden_states_139 = None
        hidden_states_140 = hidden_states_137 + dropout_52
        hidden_states_137 = dropout_52 = None
        float_145 = hidden_states_140.float()
        pow_71 = float_145.pow(2)
        mean_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_212 = mean_70 + 1e-06
        mean_70 = None
        rsqrt_70 = torch.rsqrt(add_212)
        add_212 = None
        output_140 = float_145 * rsqrt_70
        float_145 = rsqrt_70 = None
        float_146 = (
            l_self_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_213 = 1.0 + float_146
        float_146 = None
        output_141 = output_140 * add_213
        output_140 = add_213 = None
        hidden_states_141 = output_141.type_as(hidden_states_140)
        output_141 = None
        linear_123 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_17 = torch._C._nn.gelu(linear_123, approximate="tanh")
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_141 = l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_142 = gelu_17 * linear_124
        gelu_17 = linear_124 = None
        hidden_states_143 = torch.nn.functional.dropout(
            hidden_states_142, 0.0, False, False
        )
        hidden_states_142 = None
        down_proj_17 = torch._C._nn.linear(
            hidden_states_143,
            l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_143 = l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_147 = down_proj_17.float()
        pow_72 = float_147.pow(2)
        mean_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_214 = mean_71 + 1e-06
        mean_71 = None
        rsqrt_71 = torch.rsqrt(add_214)
        add_214 = None
        output_142 = float_147 * rsqrt_71
        float_147 = rsqrt_71 = None
        float_148 = (
            l_self_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_215 = 1.0 + float_148
        float_148 = None
        output_143 = output_142 * add_215
        output_142 = add_215 = None
        hidden_states_144 = output_143.type_as(down_proj_17)
        output_143 = down_proj_17 = None
        dropout_54 = torch.nn.functional.dropout(hidden_states_144, 0.0, False, False)
        hidden_states_144 = None
        hidden_states_145 = hidden_states_140 + dropout_54
        hidden_states_140 = dropout_54 = None
        float_149 = hidden_states_145.float()
        pow_73 = float_149.pow(2)
        mean_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_217 = mean_72 + 1e-06
        mean_72 = None
        rsqrt_72 = torch.rsqrt(add_217)
        add_217 = None
        output_144 = float_149 * rsqrt_72
        float_149 = rsqrt_72 = None
        float_150 = (
            l_self_modules_layers_modules_18_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_18_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_218 = 1.0 + float_150
        float_150 = None
        output_145 = output_144 * add_218
        output_144 = add_218 = None
        hidden_states_146 = output_145.type_as(hidden_states_145)
        output_145 = None
        linear_126 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_54 = linear_126.view((1, 10, -1, 64))
        linear_126 = None
        query_states_18 = view_54.transpose(1, 2)
        view_54 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_55 = linear_127.view((1, 10, -1, 64))
        linear_127 = None
        key_states_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_146 = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_56 = linear_128.view((1, 10, -1, 64))
        linear_128 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_239 = query_states_18 * cos_21
        x1_36 = query_states_18[(Ellipsis, slice(None, 32, None))]
        x2_36 = query_states_18[(Ellipsis, slice(32, None, None))]
        query_states_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_37 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_240 = cat_37 * sin_21
        cat_37 = None
        q_embed_18 = mul_239 + mul_240
        mul_239 = mul_240 = None
        mul_241 = key_states_18 * cos_21
        cos_21 = None
        x1_37 = key_states_18[(Ellipsis, slice(None, 32, None))]
        x2_37 = key_states_18[(Ellipsis, slice(32, None, None))]
        key_states_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_38 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_242 = cat_38 * sin_21
        cat_38 = sin_21 = None
        k_embed_18 = mul_241 + mul_242
        mul_241 = mul_242 = None
        attention_mask_20 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_18 = q_embed_18.contiguous()
        q_embed_18 = None
        key_18 = k_embed_18.contiguous()
        k_embed_18 = None
        value_18 = value_states_18.contiguous()
        value_states_18 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_20 = None
        transpose_76 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_76.contiguous()
        transpose_76 = None
        reshape_18 = attn_output_73.reshape(1, 10, -1)
        attn_output_73 = None
        attn_output_74 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_151 = attn_output_75.float()
        pow_74 = float_151.pow(2)
        mean_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_221 = mean_73 + 1e-06
        mean_73 = None
        rsqrt_73 = torch.rsqrt(add_221)
        add_221 = None
        output_146 = float_151 * rsqrt_73
        float_151 = rsqrt_73 = None
        float_152 = (
            l_self_modules_layers_modules_18_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_18_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_222 = 1.0 + float_152
        float_152 = None
        output_147 = output_146 * add_222
        output_146 = add_222 = None
        hidden_states_147 = output_147.type_as(attn_output_75)
        output_147 = attn_output_75 = None
        dropout_55 = torch.nn.functional.dropout(hidden_states_147, 0.0, False, False)
        hidden_states_147 = None
        hidden_states_148 = hidden_states_145 + dropout_55
        hidden_states_145 = dropout_55 = None
        float_153 = hidden_states_148.float()
        pow_75 = float_153.pow(2)
        mean_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_224 = mean_74 + 1e-06
        mean_74 = None
        rsqrt_74 = torch.rsqrt(add_224)
        add_224 = None
        output_148 = float_153 * rsqrt_74
        float_153 = rsqrt_74 = None
        float_154 = (
            l_self_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_225 = 1.0 + float_154
        float_154 = None
        output_149 = output_148 * add_225
        output_148 = add_225 = None
        hidden_states_149 = output_149.type_as(hidden_states_148)
        output_149 = None
        linear_130 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_18 = torch._C._nn.gelu(linear_130, approximate="tanh")
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_149 = l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_150 = gelu_18 * linear_131
        gelu_18 = linear_131 = None
        hidden_states_151 = torch.nn.functional.dropout(
            hidden_states_150, 0.0, False, False
        )
        hidden_states_150 = None
        down_proj_18 = torch._C._nn.linear(
            hidden_states_151,
            l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_151 = l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_155 = down_proj_18.float()
        pow_76 = float_155.pow(2)
        mean_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_226 = mean_75 + 1e-06
        mean_75 = None
        rsqrt_75 = torch.rsqrt(add_226)
        add_226 = None
        output_150 = float_155 * rsqrt_75
        float_155 = rsqrt_75 = None
        float_156 = (
            l_self_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_227 = 1.0 + float_156
        float_156 = None
        output_151 = output_150 * add_227
        output_150 = add_227 = None
        hidden_states_152 = output_151.type_as(down_proj_18)
        output_151 = down_proj_18 = None
        dropout_57 = torch.nn.functional.dropout(hidden_states_152, 0.0, False, False)
        hidden_states_152 = None
        hidden_states_153 = hidden_states_148 + dropout_57
        hidden_states_148 = dropout_57 = None
        float_157 = hidden_states_153.float()
        pow_77 = float_157.pow(2)
        mean_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_229 = mean_76 + 1e-06
        mean_76 = None
        rsqrt_76 = torch.rsqrt(add_229)
        add_229 = None
        output_152 = float_157 * rsqrt_76
        float_157 = rsqrt_76 = None
        float_158 = (
            l_self_modules_layers_modules_19_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_19_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_230 = 1.0 + float_158
        float_158 = None
        output_153 = output_152 * add_230
        output_152 = add_230 = None
        hidden_states_154 = output_153.type_as(hidden_states_153)
        output_153 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_57 = linear_133.view((1, 10, -1, 64))
        linear_133 = None
        query_states_19 = view_57.transpose(1, 2)
        view_57 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_58 = linear_134.view((1, 10, -1, 64))
        linear_134 = None
        key_states_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_154,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_154 = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_59 = linear_135.view((1, 10, -1, 64))
        linear_135 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_252 = query_states_19 * cos_22
        x1_38 = query_states_19[(Ellipsis, slice(None, 32, None))]
        x2_38 = query_states_19[(Ellipsis, slice(32, None, None))]
        query_states_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_39 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_253 = cat_39 * sin_22
        cat_39 = None
        q_embed_19 = mul_252 + mul_253
        mul_252 = mul_253 = None
        mul_254 = key_states_19 * cos_22
        cos_22 = None
        x1_39 = key_states_19[(Ellipsis, slice(None, 32, None))]
        x2_39 = key_states_19[(Ellipsis, slice(32, None, None))]
        key_states_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_40 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_255 = cat_40 * sin_22
        cat_40 = sin_22 = None
        k_embed_19 = mul_254 + mul_255
        mul_254 = mul_255 = None
        attention_mask_21 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_19 = q_embed_19.contiguous()
        q_embed_19 = None
        key_19 = k_embed_19.contiguous()
        k_embed_19 = None
        value_19 = value_states_19.contiguous()
        value_states_19 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_21 = None
        transpose_80 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_80.contiguous()
        transpose_80 = None
        reshape_19 = attn_output_77.reshape(1, 10, -1)
        attn_output_77 = None
        attn_output_78 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_159 = attn_output_79.float()
        pow_78 = float_159.pow(2)
        mean_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_233 = mean_77 + 1e-06
        mean_77 = None
        rsqrt_77 = torch.rsqrt(add_233)
        add_233 = None
        output_154 = float_159 * rsqrt_77
        float_159 = rsqrt_77 = None
        float_160 = (
            l_self_modules_layers_modules_19_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_19_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_234 = 1.0 + float_160
        float_160 = None
        output_155 = output_154 * add_234
        output_154 = add_234 = None
        hidden_states_155 = output_155.type_as(attn_output_79)
        output_155 = attn_output_79 = None
        dropout_58 = torch.nn.functional.dropout(hidden_states_155, 0.0, False, False)
        hidden_states_155 = None
        hidden_states_156 = hidden_states_153 + dropout_58
        hidden_states_153 = dropout_58 = None
        float_161 = hidden_states_156.float()
        pow_79 = float_161.pow(2)
        mean_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        add_236 = mean_78 + 1e-06
        mean_78 = None
        rsqrt_78 = torch.rsqrt(add_236)
        add_236 = None
        output_156 = float_161 * rsqrt_78
        float_161 = rsqrt_78 = None
        float_162 = (
            l_self_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_237 = 1.0 + float_162
        float_162 = None
        output_157 = output_156 * add_237
        output_156 = add_237 = None
        hidden_states_157 = output_157.type_as(hidden_states_156)
        output_157 = None
        linear_137 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_19 = torch._C._nn.gelu(linear_137, approximate="tanh")
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_157 = l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_158 = gelu_19 * linear_138
        gelu_19 = linear_138 = None
        hidden_states_159 = torch.nn.functional.dropout(
            hidden_states_158, 0.0, False, False
        )
        hidden_states_158 = None
        down_proj_19 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_159 = l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_163 = down_proj_19.float()
        pow_80 = float_163.pow(2)
        mean_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_238 = mean_79 + 1e-06
        mean_79 = None
        rsqrt_79 = torch.rsqrt(add_238)
        add_238 = None
        output_158 = float_163 * rsqrt_79
        float_163 = rsqrt_79 = None
        float_164 = (
            l_self_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_239 = 1.0 + float_164
        float_164 = None
        output_159 = output_158 * add_239
        output_158 = add_239 = None
        hidden_states_160 = output_159.type_as(down_proj_19)
        output_159 = down_proj_19 = None
        dropout_60 = torch.nn.functional.dropout(hidden_states_160, 0.0, False, False)
        hidden_states_160 = None
        hidden_states_161 = hidden_states_156 + dropout_60
        hidden_states_156 = dropout_60 = None
        float_165 = hidden_states_161.float()
        pow_81 = float_165.pow(2)
        mean_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_241 = mean_80 + 1e-06
        mean_80 = None
        rsqrt_80 = torch.rsqrt(add_241)
        add_241 = None
        output_160 = float_165 * rsqrt_80
        float_165 = rsqrt_80 = None
        float_166 = (
            l_self_modules_layers_modules_20_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_20_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_242 = 1.0 + float_166
        float_166 = None
        output_161 = output_160 * add_242
        output_160 = add_242 = None
        hidden_states_162 = output_161.type_as(hidden_states_161)
        output_161 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_60 = linear_140.view((1, 10, -1, 64))
        linear_140 = None
        query_states_20 = view_60.transpose(1, 2)
        view_60 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_61 = linear_141.view((1, 10, -1, 64))
        linear_141 = None
        key_states_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_162 = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_62 = linear_142.view((1, 10, -1, 64))
        linear_142 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_265 = query_states_20 * cos_23
        x1_40 = query_states_20[(Ellipsis, slice(None, 32, None))]
        x2_40 = query_states_20[(Ellipsis, slice(32, None, None))]
        query_states_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_41 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_266 = cat_41 * sin_23
        cat_41 = None
        q_embed_20 = mul_265 + mul_266
        mul_265 = mul_266 = None
        mul_267 = key_states_20 * cos_23
        cos_23 = None
        x1_41 = key_states_20[(Ellipsis, slice(None, 32, None))]
        x2_41 = key_states_20[(Ellipsis, slice(32, None, None))]
        key_states_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_42 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_268 = cat_42 * sin_23
        cat_42 = sin_23 = None
        k_embed_20 = mul_267 + mul_268
        mul_267 = mul_268 = None
        attention_mask_22 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_20 = q_embed_20.contiguous()
        q_embed_20 = None
        key_20 = k_embed_20.contiguous()
        k_embed_20 = None
        value_20 = value_states_20.contiguous()
        value_states_20 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_22 = None
        transpose_84 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_84.contiguous()
        transpose_84 = None
        reshape_20 = attn_output_81.reshape(1, 10, -1)
        attn_output_81 = None
        attn_output_82 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_167 = attn_output_83.float()
        pow_82 = float_167.pow(2)
        mean_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_245 = mean_81 + 1e-06
        mean_81 = None
        rsqrt_81 = torch.rsqrt(add_245)
        add_245 = None
        output_162 = float_167 * rsqrt_81
        float_167 = rsqrt_81 = None
        float_168 = (
            l_self_modules_layers_modules_20_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_20_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_246 = 1.0 + float_168
        float_168 = None
        output_163 = output_162 * add_246
        output_162 = add_246 = None
        hidden_states_163 = output_163.type_as(attn_output_83)
        output_163 = attn_output_83 = None
        dropout_61 = torch.nn.functional.dropout(hidden_states_163, 0.0, False, False)
        hidden_states_163 = None
        hidden_states_164 = hidden_states_161 + dropout_61
        hidden_states_161 = dropout_61 = None
        float_169 = hidden_states_164.float()
        pow_83 = float_169.pow(2)
        mean_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        add_248 = mean_82 + 1e-06
        mean_82 = None
        rsqrt_82 = torch.rsqrt(add_248)
        add_248 = None
        output_164 = float_169 * rsqrt_82
        float_169 = rsqrt_82 = None
        float_170 = (
            l_self_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_249 = 1.0 + float_170
        float_170 = None
        output_165 = output_164 * add_249
        output_164 = add_249 = None
        hidden_states_165 = output_165.type_as(hidden_states_164)
        output_165 = None
        linear_144 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_20 = torch._C._nn.gelu(linear_144, approximate="tanh")
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_165 = l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_166 = gelu_20 * linear_145
        gelu_20 = linear_145 = None
        hidden_states_167 = torch.nn.functional.dropout(
            hidden_states_166, 0.0, False, False
        )
        hidden_states_166 = None
        down_proj_20 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_167 = l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_171 = down_proj_20.float()
        pow_84 = float_171.pow(2)
        mean_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_250 = mean_83 + 1e-06
        mean_83 = None
        rsqrt_83 = torch.rsqrt(add_250)
        add_250 = None
        output_166 = float_171 * rsqrt_83
        float_171 = rsqrt_83 = None
        float_172 = (
            l_self_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_251 = 1.0 + float_172
        float_172 = None
        output_167 = output_166 * add_251
        output_166 = add_251 = None
        hidden_states_168 = output_167.type_as(down_proj_20)
        output_167 = down_proj_20 = None
        dropout_63 = torch.nn.functional.dropout(hidden_states_168, 0.0, False, False)
        hidden_states_168 = None
        hidden_states_169 = hidden_states_164 + dropout_63
        hidden_states_164 = dropout_63 = None
        float_173 = hidden_states_169.float()
        pow_85 = float_173.pow(2)
        mean_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_253 = mean_84 + 1e-06
        mean_84 = None
        rsqrt_84 = torch.rsqrt(add_253)
        add_253 = None
        output_168 = float_173 * rsqrt_84
        float_173 = rsqrt_84 = None
        float_174 = (
            l_self_modules_layers_modules_21_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_21_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_254 = 1.0 + float_174
        float_174 = None
        output_169 = output_168 * add_254
        output_168 = add_254 = None
        hidden_states_170 = output_169.type_as(hidden_states_169)
        output_169 = None
        linear_147 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_63 = linear_147.view((1, 10, -1, 64))
        linear_147 = None
        query_states_21 = view_63.transpose(1, 2)
        view_63 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_64 = linear_148.view((1, 10, -1, 64))
        linear_148 = None
        key_states_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_170 = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_65 = linear_149.view((1, 10, -1, 64))
        linear_149 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_278 = query_states_21 * cos_24
        x1_42 = query_states_21[(Ellipsis, slice(None, 32, None))]
        x2_42 = query_states_21[(Ellipsis, slice(32, None, None))]
        query_states_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_43 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_279 = cat_43 * sin_24
        cat_43 = None
        q_embed_21 = mul_278 + mul_279
        mul_278 = mul_279 = None
        mul_280 = key_states_21 * cos_24
        cos_24 = None
        x1_43 = key_states_21[(Ellipsis, slice(None, 32, None))]
        x2_43 = key_states_21[(Ellipsis, slice(32, None, None))]
        key_states_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_44 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_281 = cat_44 * sin_24
        cat_44 = sin_24 = None
        k_embed_21 = mul_280 + mul_281
        mul_280 = mul_281 = None
        attention_mask_23 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        query_21 = q_embed_21.contiguous()
        q_embed_21 = None
        key_21 = k_embed_21.contiguous()
        k_embed_21 = None
        value_21 = value_states_21.contiguous()
        value_states_21 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_23 = None
        transpose_88 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_88.contiguous()
        transpose_88 = None
        reshape_21 = attn_output_85.reshape(1, 10, -1)
        attn_output_85 = None
        attn_output_86 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_175 = attn_output_87.float()
        pow_86 = float_175.pow(2)
        mean_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_257 = mean_85 + 1e-06
        mean_85 = None
        rsqrt_85 = torch.rsqrt(add_257)
        add_257 = None
        output_170 = float_175 * rsqrt_85
        float_175 = rsqrt_85 = None
        float_176 = (
            l_self_modules_layers_modules_21_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_21_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_258 = 1.0 + float_176
        float_176 = None
        output_171 = output_170 * add_258
        output_170 = add_258 = None
        hidden_states_171 = output_171.type_as(attn_output_87)
        output_171 = attn_output_87 = None
        dropout_64 = torch.nn.functional.dropout(hidden_states_171, 0.0, False, False)
        hidden_states_171 = None
        hidden_states_172 = hidden_states_169 + dropout_64
        hidden_states_169 = dropout_64 = None
        float_177 = hidden_states_172.float()
        pow_87 = float_177.pow(2)
        mean_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        add_260 = mean_86 + 1e-06
        mean_86 = None
        rsqrt_86 = torch.rsqrt(add_260)
        add_260 = None
        output_172 = float_177 * rsqrt_86
        float_177 = rsqrt_86 = None
        float_178 = (
            l_self_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_261 = 1.0 + float_178
        float_178 = None
        output_173 = output_172 * add_261
        output_172 = add_261 = None
        hidden_states_173 = output_173.type_as(hidden_states_172)
        output_173 = None
        linear_151 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_21 = torch._C._nn.gelu(linear_151, approximate="tanh")
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_173 = l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_174 = gelu_21 * linear_152
        gelu_21 = linear_152 = None
        hidden_states_175 = torch.nn.functional.dropout(
            hidden_states_174, 0.0, False, False
        )
        hidden_states_174 = None
        down_proj_21 = torch._C._nn.linear(
            hidden_states_175,
            l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_175 = l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_179 = down_proj_21.float()
        pow_88 = float_179.pow(2)
        mean_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_262 = mean_87 + 1e-06
        mean_87 = None
        rsqrt_87 = torch.rsqrt(add_262)
        add_262 = None
        output_174 = float_179 * rsqrt_87
        float_179 = rsqrt_87 = None
        float_180 = (
            l_self_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_263 = 1.0 + float_180
        float_180 = None
        output_175 = output_174 * add_263
        output_174 = add_263 = None
        hidden_states_176 = output_175.type_as(down_proj_21)
        output_175 = down_proj_21 = None
        dropout_66 = torch.nn.functional.dropout(hidden_states_176, 0.0, False, False)
        hidden_states_176 = None
        hidden_states_177 = hidden_states_172 + dropout_66
        hidden_states_172 = dropout_66 = None
        float_181 = hidden_states_177.float()
        pow_89 = float_181.pow(2)
        mean_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_265 = mean_88 + 1e-06
        mean_88 = None
        rsqrt_88 = torch.rsqrt(add_265)
        add_265 = None
        output_176 = float_181 * rsqrt_88
        float_181 = rsqrt_88 = None
        float_182 = (
            l_self_modules_layers_modules_22_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_22_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_266 = 1.0 + float_182
        float_182 = None
        output_177 = output_176 * add_266
        output_176 = add_266 = None
        hidden_states_178 = output_177.type_as(hidden_states_177)
        output_177 = None
        linear_154 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_66 = linear_154.view((1, 10, -1, 64))
        linear_154 = None
        query_states_22 = view_66.transpose(1, 2)
        view_66 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_67 = linear_155.view((1, 10, -1, 64))
        linear_155 = None
        key_states_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_178 = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_68 = linear_156.view((1, 10, -1, 64))
        linear_156 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_291 = query_states_22 * cos_25
        x1_44 = query_states_22[(Ellipsis, slice(None, 32, None))]
        x2_44 = query_states_22[(Ellipsis, slice(32, None, None))]
        query_states_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_45 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_292 = cat_45 * sin_25
        cat_45 = None
        q_embed_22 = mul_291 + mul_292
        mul_291 = mul_292 = None
        mul_293 = key_states_22 * cos_25
        cos_25 = None
        x1_45 = key_states_22[(Ellipsis, slice(None, 32, None))]
        x2_45 = key_states_22[(Ellipsis, slice(32, None, None))]
        key_states_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_46 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_294 = cat_46 * sin_25
        cat_46 = sin_25 = None
        k_embed_22 = mul_293 + mul_294
        mul_293 = mul_294 = None
        attention_mask_24 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        causal_mask_1 = None
        query_22 = q_embed_22.contiguous()
        q_embed_22 = None
        key_22 = k_embed_22.contiguous()
        k_embed_22 = None
        value_22 = value_states_22.contiguous()
        value_states_22 = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_24 = None
        transpose_92 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_92.contiguous()
        transpose_92 = None
        reshape_22 = attn_output_89.reshape(1, 10, -1)
        attn_output_89 = None
        attn_output_90 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_183 = attn_output_91.float()
        pow_90 = float_183.pow(2)
        mean_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_269 = mean_89 + 1e-06
        mean_89 = None
        rsqrt_89 = torch.rsqrt(add_269)
        add_269 = None
        output_178 = float_183 * rsqrt_89
        float_183 = rsqrt_89 = None
        float_184 = (
            l_self_modules_layers_modules_22_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_22_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_270 = 1.0 + float_184
        float_184 = None
        output_179 = output_178 * add_270
        output_178 = add_270 = None
        hidden_states_179 = output_179.type_as(attn_output_91)
        output_179 = attn_output_91 = None
        dropout_67 = torch.nn.functional.dropout(hidden_states_179, 0.0, False, False)
        hidden_states_179 = None
        hidden_states_180 = hidden_states_177 + dropout_67
        hidden_states_177 = dropout_67 = None
        float_185 = hidden_states_180.float()
        pow_91 = float_185.pow(2)
        mean_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        add_272 = mean_90 + 1e-06
        mean_90 = None
        rsqrt_90 = torch.rsqrt(add_272)
        add_272 = None
        output_180 = float_185 * rsqrt_90
        float_185 = rsqrt_90 = None
        float_186 = (
            l_self_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_273 = 1.0 + float_186
        float_186 = None
        output_181 = output_180 * add_273
        output_180 = add_273 = None
        hidden_states_181 = output_181.type_as(hidden_states_180)
        output_181 = None
        linear_158 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_22 = torch._C._nn.gelu(linear_158, approximate="tanh")
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_181 = l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_182 = gelu_22 * linear_159
        gelu_22 = linear_159 = None
        hidden_states_183 = torch.nn.functional.dropout(
            hidden_states_182, 0.0, False, False
        )
        hidden_states_182 = None
        down_proj_22 = torch._C._nn.linear(
            hidden_states_183,
            l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_183 = l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_187 = down_proj_22.float()
        pow_92 = float_187.pow(2)
        mean_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_274 = mean_91 + 1e-06
        mean_91 = None
        rsqrt_91 = torch.rsqrt(add_274)
        add_274 = None
        output_182 = float_187 * rsqrt_91
        float_187 = rsqrt_91 = None
        float_188 = (
            l_self_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_275 = 1.0 + float_188
        float_188 = None
        output_183 = output_182 * add_275
        output_182 = add_275 = None
        hidden_states_184 = output_183.type_as(down_proj_22)
        output_183 = down_proj_22 = None
        dropout_69 = torch.nn.functional.dropout(hidden_states_184, 0.0, False, False)
        hidden_states_184 = None
        hidden_states_185 = hidden_states_180 + dropout_69
        hidden_states_180 = dropout_69 = None
        float_189 = hidden_states_185.float()
        pow_93 = float_189.pow(2)
        mean_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_277 = mean_92 + 1e-06
        mean_92 = None
        rsqrt_92 = torch.rsqrt(add_277)
        add_277 = None
        output_184 = float_189 * rsqrt_92
        float_189 = rsqrt_92 = None
        float_190 = (
            l_self_modules_layers_modules_23_modules_pre_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_23_modules_pre_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_278 = 1.0 + float_190
        float_190 = None
        output_185 = output_184 * add_278
        output_184 = add_278 = None
        hidden_states_186 = output_185.type_as(hidden_states_185)
        output_185 = None
        linear_161 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_69 = linear_161.view((1, 10, -1, 64))
        linear_161 = None
        query_states_23 = view_69.transpose(1, 2)
        view_69 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_70 = linear_162.view((1, 10, -1, 64))
        linear_162 = None
        key_states_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_186 = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_71 = linear_163.view((1, 10, -1, 64))
        linear_163 = None
        value_states_23 = view_71.transpose(1, 2)
        view_71 = None
        cos_26 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_26 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_304 = query_states_23 * cos_26
        x1_46 = query_states_23[(Ellipsis, slice(None, 32, None))]
        x2_46 = query_states_23[(Ellipsis, slice(32, None, None))]
        query_states_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_47 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_305 = cat_47 * sin_26
        cat_47 = None
        q_embed_23 = mul_304 + mul_305
        mul_304 = mul_305 = None
        mul_306 = key_states_23 * cos_26
        cos_26 = None
        x1_47 = key_states_23[(Ellipsis, slice(None, 32, None))]
        x2_47 = key_states_23[(Ellipsis, slice(32, None, None))]
        key_states_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_48 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_307 = cat_48 * sin_26
        cat_48 = sin_26 = None
        k_embed_23 = mul_306 + mul_307
        mul_306 = mul_307 = None
        attention_mask_25 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 10, None),
            )
        ]
        causal_mask = None
        query_23 = q_embed_23.contiguous()
        q_embed_23 = None
        key_23 = k_embed_23.contiguous()
        k_embed_23 = None
        value_23 = value_states_23.contiguous()
        value_states_23 = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.125,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_25 = None
        transpose_96 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_96.contiguous()
        transpose_96 = None
        reshape_23 = attn_output_93.reshape(1, 10, -1)
        attn_output_93 = None
        attn_output_94 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        float_191 = attn_output_95.float()
        pow_94 = float_191.pow(2)
        mean_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_281 = mean_93 + 1e-06
        mean_93 = None
        rsqrt_93 = torch.rsqrt(add_281)
        add_281 = None
        output_186 = float_191 * rsqrt_93
        float_191 = rsqrt_93 = None
        float_192 = (
            l_self_modules_layers_modules_23_modules_post_self_attn_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_23_modules_post_self_attn_layernorm_parameters_weight_ = (
            None
        )
        add_282 = 1.0 + float_192
        float_192 = None
        output_187 = output_186 * add_282
        output_186 = add_282 = None
        hidden_states_187 = output_187.type_as(attn_output_95)
        output_187 = attn_output_95 = None
        dropout_70 = torch.nn.functional.dropout(hidden_states_187, 0.0, False, False)
        hidden_states_187 = None
        hidden_states_188 = hidden_states_185 + dropout_70
        hidden_states_185 = dropout_70 = None
        float_193 = hidden_states_188.float()
        pow_95 = float_193.pow(2)
        mean_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        add_284 = mean_94 + 1e-06
        mean_94 = None
        rsqrt_94 = torch.rsqrt(add_284)
        add_284 = None
        output_188 = float_193 * rsqrt_94
        float_193 = rsqrt_94 = None
        float_194 = (
            l_self_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_285 = 1.0 + float_194
        float_194 = None
        output_189 = output_188 * add_285
        output_188 = add_285 = None
        hidden_states_189 = output_189.type_as(hidden_states_188)
        output_189 = None
        linear_165 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_23 = torch._C._nn.gelu(linear_165, approximate="tanh")
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_189 = l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        hidden_states_190 = gelu_23 * linear_166
        gelu_23 = linear_166 = None
        hidden_states_191 = torch.nn.functional.dropout(
            hidden_states_190, 0.0, False, False
        )
        hidden_states_190 = None
        down_proj_23 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        hidden_states_191 = l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_195 = down_proj_23.float()
        pow_96 = float_195.pow(2)
        mean_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_286 = mean_95 + 1e-06
        mean_95 = None
        rsqrt_95 = torch.rsqrt(add_286)
        add_286 = None
        output_190 = float_195 * rsqrt_95
        float_195 = rsqrt_95 = None
        float_196 = (
            l_self_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_287 = 1.0 + float_196
        float_196 = None
        output_191 = output_190 * add_287
        output_190 = add_287 = None
        hidden_states_192 = output_191.type_as(down_proj_23)
        output_191 = down_proj_23 = None
        dropout_72 = torch.nn.functional.dropout(hidden_states_192, 0.0, False, False)
        hidden_states_192 = None
        hidden_states_193 = hidden_states_188 + dropout_72
        hidden_states_188 = dropout_72 = None
        float_197 = hidden_states_193.float()
        pow_97 = float_197.pow(2)
        mean_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_289 = mean_96 + 1e-06
        mean_96 = None
        rsqrt_96 = torch.rsqrt(add_289)
        add_289 = None
        output_192 = float_197 * rsqrt_96
        float_197 = rsqrt_96 = None
        float_198 = l_self_modules_norm_parameters_weight_.float()
        l_self_modules_norm_parameters_weight_ = None
        add_290 = 1.0 + float_198
        float_198 = None
        output_193 = output_192 * add_290
        output_192 = add_290 = None
        hidden_states_194 = output_193.type_as(hidden_states_193)
        output_193 = hidden_states_193 = None
        hidden_states_195 = torch.nn.functional.dropout(
            hidden_states_194, 0.0, False, False
        )
        hidden_states_194 = None
        return (hidden_states_195,)
