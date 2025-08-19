import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_inputs_embeds_ = L_inputs_embeds_
        l_attention_mask_ = L_attention_mask_
        l_self_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_32_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_33_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_34_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        cache_position = torch.arange(0, 2, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_attention_mask_ = None
        mask_indices = torch.arange(2, device=device(type="cuda", index=0))
        mask_indices += 0
        mask_indices_1 = mask_indices
        mask_indices = None
        local_padding_mask = attention_mask[(slice(None, None, None), mask_indices_1)]
        attention_mask = mask_indices_1 = None
        kv_arange = torch.arange(2, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        reshaped_cache_position = cache_position.view(-1, 1)
        cache_position = None
        causal_mask = kv_arange_1 <= reshaped_cache_position
        kv_arange_1 = reshaped_cache_position = None
        getitem_1 = causal_mask[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask = None
        causal_mask_1 = getitem_1.expand(1, -1, -1, -1)
        getitem_1 = None
        getitem_2 = local_padding_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        local_padding_mask = None
        causal_mask_2 = causal_mask_1 * getitem_2
        causal_mask_1 = getitem_2 = None
        _set_grad_enabled = torch._C._set_grad_enabled(False)
        _set_grad_enabled = None
        getitem_3 = l_self_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem_3.float()
        getitem_3 = None
        expand_1 = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand_1.to(device(type="cuda", index=0))
        expand_1 = None
        getitem_4 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_4.float()
        getitem_4 = None
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
        cos_2 = cos_1.to(dtype=torch.bfloat16)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.bfloat16)
        sin_1 = None
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        _set_grad_enabled_1 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        hidden_states = l_inputs_embeds_.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-06
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        hidden_states = rsqrt = None
        to_5 = hidden_states_1.to(torch.bfloat16)
        hidden_states_1 = None
        hidden_states_2 = (
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
            * to_5
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            to_5
        ) = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_1 = linear.view((1, 2, -1, 128))
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_2 = linear_1.view((1, 2, -1, 128))
        linear_1 = None
        key_states = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_3 = linear_2.view((1, 2, -1, 128))
        linear_2 = None
        value_states = view_3.transpose(1, 2)
        view_3 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_5 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 64, None))]
        x2 = query_states[(Ellipsis, slice(64, None, None))]
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
        x1_1 = key_states[(Ellipsis, slice(None, 64, None))]
        x2_1 = key_states[(Ellipsis, slice(64, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_8 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_7 + mul_8
        mul_7 = mul_8 = None
        getitem_9 = k_embed[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_3 = getitem_9.expand(1, 2, 8, 2, 128)
        getitem_9 = None
        key = hidden_states_3.reshape(1, 16, 2, 128)
        hidden_states_3 = None
        getitem_10 = value_states[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_4 = getitem_10.expand(1, 2, 8, 2, 128)
        getitem_10 = None
        value = hidden_states_4.reshape(1, 16, 2, 128)
        hidden_states_4 = None
        attention_mask_1 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query = q_embed.contiguous()
        q_embed = None
        key_1 = key.contiguous()
        key = None
        value_1 = value.contiguous()
        value = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query = key_1 = value_1 = attention_mask_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape_2 = attn_output_1.reshape(1, 2, -1)
        attn_output_1 = None
        attn_output_2 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_5 = l_inputs_embeds_ + attn_output_3
        l_inputs_embeds_ = attn_output_3 = None
        hidden_states_6 = hidden_states_5.to(torch.float32)
        pow_2 = hidden_states_6.pow(2)
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_4 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_4)
        add_4 = None
        hidden_states_7 = hidden_states_6 * rsqrt_1
        hidden_states_6 = rsqrt_1 = None
        to_7 = hidden_states_7.to(torch.bfloat16)
        hidden_states_7 = None
        hidden_states_8 = (
            l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_7
        )
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_7
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_8 = l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_11 = silu * linear_5
        silu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_11,
            l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_11 = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_9 = hidden_states_5 + down_proj
        hidden_states_5 = down_proj = None
        hidden_states_10 = hidden_states_9.to(torch.float32)
        pow_3 = hidden_states_10.pow(2)
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_6 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_11 = hidden_states_10 * rsqrt_2
        hidden_states_10 = rsqrt_2 = None
        to_9 = hidden_states_11.to(torch.bfloat16)
        hidden_states_11 = None
        hidden_states_12 = (
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_9
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_4 = linear_7.view((1, 2, -1, 128))
        linear_7 = None
        query_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_5 = linear_8.view((1, 2, -1, 128))
        linear_8 = None
        key_states_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_6 = linear_9.view((1, 2, -1, 128))
        linear_9 = None
        value_states_1 = view_6.transpose(1, 2)
        view_6 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_14 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 64, None))]
        x2_2 = query_states_1[(Ellipsis, slice(64, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_15 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_14 + mul_15
        mul_14 = mul_15 = None
        mul_16 = key_states_1 * cos_4
        cos_4 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 64, None))]
        x2_3 = key_states_1[(Ellipsis, slice(64, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_17 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_16 + mul_17
        mul_16 = mul_17 = None
        getitem_16 = k_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_13 = getitem_16.expand(1, 2, 8, 2, 128)
        getitem_16 = None
        key_2 = hidden_states_13.reshape(1, 16, 2, 128)
        hidden_states_13 = None
        getitem_17 = value_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_14 = getitem_17.expand(1, 2, 8, 2, 128)
        getitem_17 = None
        value_2 = hidden_states_14.reshape(1, 16, 2, 128)
        hidden_states_14 = None
        attention_mask_2 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_1 = q_embed_1.contiguous()
        q_embed_1 = None
        key_3 = key_2.contiguous()
        key_2 = None
        value_3 = value_2.contiguous()
        value_2 = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_3,
            value_3,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_1 = key_3 = value_3 = attention_mask_2 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_5 = attn_output_5.reshape(1, 2, -1)
        attn_output_5 = None
        attn_output_6 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_15 = hidden_states_9 + attn_output_7
        hidden_states_9 = attn_output_7 = None
        hidden_states_16 = hidden_states_15.to(torch.float32)
        pow_4 = hidden_states_16.pow(2)
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_17 = hidden_states_16 * rsqrt_3
        hidden_states_16 = rsqrt_3 = None
        to_11 = hidden_states_17.to(torch.bfloat16)
        hidden_states_17 = None
        hidden_states_18 = (
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_11
        ) = None
        linear_11 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_18 = l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_20 = silu_1 * linear_12
        silu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_20,
            l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_20 = l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_19 = hidden_states_15 + down_proj_1
        hidden_states_15 = down_proj_1 = None
        hidden_states_20 = hidden_states_19.to(torch.float32)
        pow_5 = hidden_states_20.pow(2)
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_12 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_21 = hidden_states_20 * rsqrt_4
        hidden_states_20 = rsqrt_4 = None
        to_13 = hidden_states_21.to(torch.bfloat16)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_13
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_7 = linear_14.view((1, 2, -1, 128))
        linear_14 = None
        query_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_8 = linear_15.view((1, 2, -1, 128))
        linear_15 = None
        key_states_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_9 = linear_16.view((1, 2, -1, 128))
        linear_16 = None
        value_states_2 = view_9.transpose(1, 2)
        view_9 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_23 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 64, None))]
        x2_4 = query_states_2[(Ellipsis, slice(64, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_24 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_23 + mul_24
        mul_23 = mul_24 = None
        mul_25 = key_states_2 * cos_5
        cos_5 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 64, None))]
        x2_5 = key_states_2[(Ellipsis, slice(64, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_26 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_25 + mul_26
        mul_25 = mul_26 = None
        getitem_23 = k_embed_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_23 = getitem_23.expand(1, 2, 8, 2, 128)
        getitem_23 = None
        key_4 = hidden_states_23.reshape(1, 16, 2, 128)
        hidden_states_23 = None
        getitem_24 = value_states_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_24 = getitem_24.expand(1, 2, 8, 2, 128)
        getitem_24 = None
        value_4 = hidden_states_24.reshape(1, 16, 2, 128)
        hidden_states_24 = None
        attention_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_2 = q_embed_2.contiguous()
        q_embed_2 = None
        key_5 = key_4.contiguous()
        key_4 = None
        value_5 = value_4.contiguous()
        value_4 = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_5,
            value_5,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_2 = key_5 = value_5 = attention_mask_3 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_8 = attn_output_9.reshape(1, 2, -1)
        attn_output_9 = None
        attn_output_10 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_25 = hidden_states_19 + attn_output_11
        hidden_states_19 = attn_output_11 = None
        hidden_states_26 = hidden_states_25.to(torch.float32)
        pow_6 = hidden_states_26.pow(2)
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_16 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_27 = hidden_states_26 * rsqrt_5
        hidden_states_26 = rsqrt_5 = None
        to_15 = hidden_states_27.to(torch.bfloat16)
        hidden_states_27 = None
        hidden_states_28 = (
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_28 = l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_29 = silu_2 * linear_19
        silu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_29,
            l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_29 = l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_29 = hidden_states_25 + down_proj_2
        hidden_states_25 = down_proj_2 = None
        hidden_states_30 = hidden_states_29.to(torch.float32)
        pow_7 = hidden_states_30.pow(2)
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_18 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_31 = hidden_states_30 * rsqrt_6
        hidden_states_30 = rsqrt_6 = None
        to_17 = hidden_states_31.to(torch.bfloat16)
        hidden_states_31 = None
        hidden_states_32 = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_17
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_10 = linear_21.view((1, 2, -1, 128))
        linear_21 = None
        query_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_11 = linear_22.view((1, 2, -1, 128))
        linear_22 = None
        key_states_3 = view_11.transpose(1, 2)
        view_11 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_12 = linear_23.view((1, 2, -1, 128))
        linear_23 = None
        value_states_3 = view_12.transpose(1, 2)
        view_12 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_32 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 64, None))]
        x2_6 = query_states_3[(Ellipsis, slice(64, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_33 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_32 + mul_33
        mul_32 = mul_33 = None
        mul_34 = key_states_3 * cos_6
        cos_6 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 64, None))]
        x2_7 = key_states_3[(Ellipsis, slice(64, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_35 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_34 + mul_35
        mul_34 = mul_35 = None
        getitem_30 = k_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_33 = getitem_30.expand(1, 2, 8, 2, 128)
        getitem_30 = None
        key_6 = hidden_states_33.reshape(1, 16, 2, 128)
        hidden_states_33 = None
        getitem_31 = value_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_34 = getitem_31.expand(1, 2, 8, 2, 128)
        getitem_31 = None
        value_6 = hidden_states_34.reshape(1, 16, 2, 128)
        hidden_states_34 = None
        attention_mask_4 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_3 = q_embed_3.contiguous()
        q_embed_3 = None
        key_7 = key_6.contiguous()
        key_6 = None
        value_7 = value_6.contiguous()
        value_6 = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_7,
            value_7,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_3 = key_7 = value_7 = attention_mask_4 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_11 = attn_output_13.reshape(1, 2, -1)
        attn_output_13 = None
        attn_output_14 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_35 = hidden_states_29 + attn_output_15
        hidden_states_29 = attn_output_15 = None
        hidden_states_36 = hidden_states_35.to(torch.float32)
        pow_8 = hidden_states_36.pow(2)
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_22 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_37 = hidden_states_36 * rsqrt_7
        hidden_states_36 = rsqrt_7 = None
        to_19 = hidden_states_37.to(torch.bfloat16)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_19
        ) = None
        linear_25 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_38 = silu_3 * linear_26
        silu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_38,
            l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_38 = l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_39 = hidden_states_35 + down_proj_3
        hidden_states_35 = down_proj_3 = None
        hidden_states_40 = hidden_states_39.to(torch.float32)
        pow_9 = hidden_states_40.pow(2)
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_24 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_41 = hidden_states_40 * rsqrt_8
        hidden_states_40 = rsqrt_8 = None
        to_21 = hidden_states_41.to(torch.bfloat16)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_21
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_13 = linear_28.view((1, 2, -1, 128))
        linear_28 = None
        query_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_14 = linear_29.view((1, 2, -1, 128))
        linear_29 = None
        key_states_4 = view_14.transpose(1, 2)
        view_14 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_15 = linear_30.view((1, 2, -1, 128))
        linear_30 = None
        value_states_4 = view_15.transpose(1, 2)
        view_15 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_41 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 64, None))]
        x2_8 = query_states_4[(Ellipsis, slice(64, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_42 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_41 + mul_42
        mul_41 = mul_42 = None
        mul_43 = key_states_4 * cos_7
        cos_7 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 64, None))]
        x2_9 = key_states_4[(Ellipsis, slice(64, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_44 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_43 + mul_44
        mul_43 = mul_44 = None
        getitem_37 = k_embed_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_43 = getitem_37.expand(1, 2, 8, 2, 128)
        getitem_37 = None
        key_8 = hidden_states_43.reshape(1, 16, 2, 128)
        hidden_states_43 = None
        getitem_38 = value_states_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_44 = getitem_38.expand(1, 2, 8, 2, 128)
        getitem_38 = None
        value_8 = hidden_states_44.reshape(1, 16, 2, 128)
        hidden_states_44 = None
        attention_mask_5 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_4 = q_embed_4.contiguous()
        q_embed_4 = None
        key_9 = key_8.contiguous()
        key_8 = None
        value_9 = value_8.contiguous()
        value_8 = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_9,
            value_9,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_4 = key_9 = value_9 = attention_mask_5 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_14 = attn_output_17.reshape(1, 2, -1)
        attn_output_17 = None
        attn_output_18 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_45 = hidden_states_39 + attn_output_19
        hidden_states_39 = attn_output_19 = None
        hidden_states_46 = hidden_states_45.to(torch.float32)
        pow_10 = hidden_states_46.pow(2)
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_28 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_47 = hidden_states_46 * rsqrt_9
        hidden_states_46 = rsqrt_9 = None
        to_23 = hidden_states_47.to(torch.bfloat16)
        hidden_states_47 = None
        hidden_states_48 = (
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_23
        ) = None
        linear_32 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_48 = l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_47 = silu_4 * linear_33
        silu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_47,
            l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_47 = l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_49 = hidden_states_45 + down_proj_4
        hidden_states_45 = down_proj_4 = None
        hidden_states_50 = hidden_states_49.to(torch.float32)
        pow_11 = hidden_states_50.pow(2)
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_30 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_51 = hidden_states_50 * rsqrt_10
        hidden_states_50 = rsqrt_10 = None
        to_25 = hidden_states_51.to(torch.bfloat16)
        hidden_states_51 = None
        hidden_states_52 = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_25
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_16 = linear_35.view((1, 2, -1, 128))
        linear_35 = None
        query_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_17 = linear_36.view((1, 2, -1, 128))
        linear_36 = None
        key_states_5 = view_17.transpose(1, 2)
        view_17 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_18 = linear_37.view((1, 2, -1, 128))
        linear_37 = None
        value_states_5 = view_18.transpose(1, 2)
        view_18 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_50 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 64, None))]
        x2_10 = query_states_5[(Ellipsis, slice(64, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_51 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_50 + mul_51
        mul_50 = mul_51 = None
        mul_52 = key_states_5 * cos_8
        cos_8 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 64, None))]
        x2_11 = key_states_5[(Ellipsis, slice(64, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_53 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_52 + mul_53
        mul_52 = mul_53 = None
        getitem_44 = k_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_53 = getitem_44.expand(1, 2, 8, 2, 128)
        getitem_44 = None
        key_10 = hidden_states_53.reshape(1, 16, 2, 128)
        hidden_states_53 = None
        getitem_45 = value_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_54 = getitem_45.expand(1, 2, 8, 2, 128)
        getitem_45 = None
        value_10 = hidden_states_54.reshape(1, 16, 2, 128)
        hidden_states_54 = None
        attention_mask_6 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_5 = q_embed_5.contiguous()
        q_embed_5 = None
        key_11 = key_10.contiguous()
        key_10 = None
        value_11 = value_10.contiguous()
        value_10 = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_11,
            value_11,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_5 = key_11 = value_11 = attention_mask_6 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_17 = attn_output_21.reshape(1, 2, -1)
        attn_output_21 = None
        attn_output_22 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_55 = hidden_states_49 + attn_output_23
        hidden_states_49 = attn_output_23 = None
        hidden_states_56 = hidden_states_55.to(torch.float32)
        pow_12 = hidden_states_56.pow(2)
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_34 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_57 = hidden_states_56 * rsqrt_11
        hidden_states_56 = rsqrt_11 = None
        to_27 = hidden_states_57.to(torch.bfloat16)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_27
        ) = None
        linear_39 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_58 = l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_56 = silu_5 * linear_40
        silu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_56,
            l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_56 = l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_59 = hidden_states_55 + down_proj_5
        hidden_states_55 = down_proj_5 = None
        hidden_states_60 = hidden_states_59.to(torch.float32)
        pow_13 = hidden_states_60.pow(2)
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_36 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_61 = hidden_states_60 * rsqrt_12
        hidden_states_60 = rsqrt_12 = None
        to_29 = hidden_states_61.to(torch.bfloat16)
        hidden_states_61 = None
        hidden_states_62 = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_29
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_19 = linear_42.view((1, 2, -1, 128))
        linear_42 = None
        query_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_20 = linear_43.view((1, 2, -1, 128))
        linear_43 = None
        key_states_6 = view_20.transpose(1, 2)
        view_20 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_21 = linear_44.view((1, 2, -1, 128))
        linear_44 = None
        value_states_6 = view_21.transpose(1, 2)
        view_21 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_59 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 64, None))]
        x2_12 = query_states_6[(Ellipsis, slice(64, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_60 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_59 + mul_60
        mul_59 = mul_60 = None
        mul_61 = key_states_6 * cos_9
        cos_9 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 64, None))]
        x2_13 = key_states_6[(Ellipsis, slice(64, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_62 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_61 + mul_62
        mul_61 = mul_62 = None
        getitem_51 = k_embed_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_63 = getitem_51.expand(1, 2, 8, 2, 128)
        getitem_51 = None
        key_12 = hidden_states_63.reshape(1, 16, 2, 128)
        hidden_states_63 = None
        getitem_52 = value_states_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_64 = getitem_52.expand(1, 2, 8, 2, 128)
        getitem_52 = None
        value_12 = hidden_states_64.reshape(1, 16, 2, 128)
        hidden_states_64 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_6 = q_embed_6.contiguous()
        q_embed_6 = None
        key_13 = key_12.contiguous()
        key_12 = None
        value_13 = value_12.contiguous()
        value_12 = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_13,
            value_13,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_6 = key_13 = value_13 = attention_mask_7 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_20 = attn_output_25.reshape(1, 2, -1)
        attn_output_25 = None
        attn_output_26 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_65 = hidden_states_59 + attn_output_27
        hidden_states_59 = attn_output_27 = None
        hidden_states_66 = hidden_states_65.to(torch.float32)
        pow_14 = hidden_states_66.pow(2)
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_40 = variance_13 + 1e-06
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_67 = hidden_states_66 * rsqrt_13
        hidden_states_66 = rsqrt_13 = None
        to_31 = hidden_states_67.to(torch.bfloat16)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_31
        ) = None
        linear_46 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_68 = l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_65 = silu_6 * linear_47
        silu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_65,
            l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_65 = l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_69 = hidden_states_65 + down_proj_6
        hidden_states_65 = down_proj_6 = None
        hidden_states_70 = hidden_states_69.to(torch.float32)
        pow_15 = hidden_states_70.pow(2)
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_42 = variance_14 + 1e-06
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_71 = hidden_states_70 * rsqrt_14
        hidden_states_70 = rsqrt_14 = None
        to_33 = hidden_states_71.to(torch.bfloat16)
        hidden_states_71 = None
        hidden_states_72 = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_33
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_22 = linear_49.view((1, 2, -1, 128))
        linear_49 = None
        query_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_23 = linear_50.view((1, 2, -1, 128))
        linear_50 = None
        key_states_7 = view_23.transpose(1, 2)
        view_23 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_72 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_24 = linear_51.view((1, 2, -1, 128))
        linear_51 = None
        value_states_7 = view_24.transpose(1, 2)
        view_24 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_68 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 64, None))]
        x2_14 = query_states_7[(Ellipsis, slice(64, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_69 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_68 + mul_69
        mul_68 = mul_69 = None
        mul_70 = key_states_7 * cos_10
        cos_10 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 64, None))]
        x2_15 = key_states_7[(Ellipsis, slice(64, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_71 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_70 + mul_71
        mul_70 = mul_71 = None
        getitem_58 = k_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_73 = getitem_58.expand(1, 2, 8, 2, 128)
        getitem_58 = None
        key_14 = hidden_states_73.reshape(1, 16, 2, 128)
        hidden_states_73 = None
        getitem_59 = value_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_74 = getitem_59.expand(1, 2, 8, 2, 128)
        getitem_59 = None
        value_14 = hidden_states_74.reshape(1, 16, 2, 128)
        hidden_states_74 = None
        attention_mask_8 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_7 = q_embed_7.contiguous()
        q_embed_7 = None
        key_15 = key_14.contiguous()
        key_14 = None
        value_15 = value_14.contiguous()
        value_14 = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_15,
            value_15,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_7 = key_15 = value_15 = attention_mask_8 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_23 = attn_output_29.reshape(1, 2, -1)
        attn_output_29 = None
        attn_output_30 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_75 = hidden_states_69 + attn_output_31
        hidden_states_69 = attn_output_31 = None
        hidden_states_76 = hidden_states_75.to(torch.float32)
        pow_16 = hidden_states_76.pow(2)
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_46 = variance_15 + 1e-06
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_77 = hidden_states_76 * rsqrt_15
        hidden_states_76 = rsqrt_15 = None
        to_35 = hidden_states_77.to(torch.bfloat16)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_35
        ) = None
        linear_53 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_78 = l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_74 = silu_7 * linear_54
        silu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_74,
            l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_74 = l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_79 = hidden_states_75 + down_proj_7
        hidden_states_75 = down_proj_7 = None
        hidden_states_80 = hidden_states_79.to(torch.float32)
        pow_17 = hidden_states_80.pow(2)
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_48 = variance_16 + 1e-06
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_81 = hidden_states_80 * rsqrt_16
        hidden_states_80 = rsqrt_16 = None
        to_37 = hidden_states_81.to(torch.bfloat16)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_37
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_25 = linear_56.view((1, 2, -1, 128))
        linear_56 = None
        query_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_26 = linear_57.view((1, 2, -1, 128))
        linear_57 = None
        key_states_8 = view_26.transpose(1, 2)
        view_26 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_27 = linear_58.view((1, 2, -1, 128))
        linear_58 = None
        value_states_8 = view_27.transpose(1, 2)
        view_27 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_77 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 64, None))]
        x2_16 = query_states_8[(Ellipsis, slice(64, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_78 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_77 + mul_78
        mul_77 = mul_78 = None
        mul_79 = key_states_8 * cos_11
        cos_11 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 64, None))]
        x2_17 = key_states_8[(Ellipsis, slice(64, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_80 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_79 + mul_80
        mul_79 = mul_80 = None
        getitem_65 = k_embed_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_83 = getitem_65.expand(1, 2, 8, 2, 128)
        getitem_65 = None
        key_16 = hidden_states_83.reshape(1, 16, 2, 128)
        hidden_states_83 = None
        getitem_66 = value_states_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_84 = getitem_66.expand(1, 2, 8, 2, 128)
        getitem_66 = None
        value_16 = hidden_states_84.reshape(1, 16, 2, 128)
        hidden_states_84 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_8 = q_embed_8.contiguous()
        q_embed_8 = None
        key_17 = key_16.contiguous()
        key_16 = None
        value_17 = value_16.contiguous()
        value_16 = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_17,
            value_17,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_8 = key_17 = value_17 = attention_mask_9 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_26 = attn_output_33.reshape(1, 2, -1)
        attn_output_33 = None
        attn_output_34 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_85 = hidden_states_79 + attn_output_35
        hidden_states_79 = attn_output_35 = None
        hidden_states_86 = hidden_states_85.to(torch.float32)
        pow_18 = hidden_states_86.pow(2)
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_52 = variance_17 + 1e-06
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_87 = hidden_states_86 * rsqrt_17
        hidden_states_86 = rsqrt_17 = None
        to_39 = hidden_states_87.to(torch.bfloat16)
        hidden_states_87 = None
        hidden_states_88 = (
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_39
        ) = None
        linear_60 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_60, inplace=False)
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_88 = l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_83 = silu_8 * linear_61
        silu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_83,
            l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_83 = l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_89 = hidden_states_85 + down_proj_8
        hidden_states_85 = down_proj_8 = None
        hidden_states_90 = hidden_states_89.to(torch.float32)
        pow_19 = hidden_states_90.pow(2)
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_54 = variance_18 + 1e-06
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_91 = hidden_states_90 * rsqrt_18
        hidden_states_90 = rsqrt_18 = None
        to_41 = hidden_states_91.to(torch.bfloat16)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_41
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_41
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_28 = linear_63.view((1, 2, -1, 128))
        linear_63 = None
        query_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_29 = linear_64.view((1, 2, -1, 128))
        linear_64 = None
        key_states_9 = view_29.transpose(1, 2)
        view_29 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_30 = linear_65.view((1, 2, -1, 128))
        linear_65 = None
        value_states_9 = view_30.transpose(1, 2)
        view_30 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_86 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 64, None))]
        x2_18 = query_states_9[(Ellipsis, slice(64, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_87 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_86 + mul_87
        mul_86 = mul_87 = None
        mul_88 = key_states_9 * cos_12
        cos_12 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 64, None))]
        x2_19 = key_states_9[(Ellipsis, slice(64, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_89 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_88 + mul_89
        mul_88 = mul_89 = None
        getitem_72 = k_embed_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_93 = getitem_72.expand(1, 2, 8, 2, 128)
        getitem_72 = None
        key_18 = hidden_states_93.reshape(1, 16, 2, 128)
        hidden_states_93 = None
        getitem_73 = value_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_94 = getitem_73.expand(1, 2, 8, 2, 128)
        getitem_73 = None
        value_18 = hidden_states_94.reshape(1, 16, 2, 128)
        hidden_states_94 = None
        attention_mask_10 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_9 = q_embed_9.contiguous()
        q_embed_9 = None
        key_19 = key_18.contiguous()
        key_18 = None
        value_19 = value_18.contiguous()
        value_18 = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_19,
            value_19,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_9 = key_19 = value_19 = attention_mask_10 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_29 = attn_output_37.reshape(1, 2, -1)
        attn_output_37 = None
        attn_output_38 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_95 = hidden_states_89 + attn_output_39
        hidden_states_89 = attn_output_39 = None
        hidden_states_96 = hidden_states_95.to(torch.float32)
        pow_20 = hidden_states_96.pow(2)
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_58 = variance_19 + 1e-06
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_58)
        add_58 = None
        hidden_states_97 = hidden_states_96 * rsqrt_19
        hidden_states_96 = rsqrt_19 = None
        to_43 = hidden_states_97.to(torch.bfloat16)
        hidden_states_97 = None
        hidden_states_98 = (
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_43
        ) = None
        linear_67 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_67, inplace=False)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_98 = l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_92 = silu_9 * linear_68
        silu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_92,
            l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_92 = l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_99 = hidden_states_95 + down_proj_9
        hidden_states_95 = down_proj_9 = None
        hidden_states_100 = hidden_states_99.to(torch.float32)
        pow_21 = hidden_states_100.pow(2)
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_60 = variance_20 + 1e-06
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_60)
        add_60 = None
        hidden_states_101 = hidden_states_100 * rsqrt_20
        hidden_states_100 = rsqrt_20 = None
        to_45 = hidden_states_101.to(torch.bfloat16)
        hidden_states_101 = None
        hidden_states_102 = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_45
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_31 = linear_70.view((1, 2, -1, 128))
        linear_70 = None
        query_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_32 = linear_71.view((1, 2, -1, 128))
        linear_71 = None
        key_states_10 = view_32.transpose(1, 2)
        view_32 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_102 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_33 = linear_72.view((1, 2, -1, 128))
        linear_72 = None
        value_states_10 = view_33.transpose(1, 2)
        view_33 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_95 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 64, None))]
        x2_20 = query_states_10[(Ellipsis, slice(64, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_96 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_95 + mul_96
        mul_95 = mul_96 = None
        mul_97 = key_states_10 * cos_13
        cos_13 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 64, None))]
        x2_21 = key_states_10[(Ellipsis, slice(64, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_98 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_97 + mul_98
        mul_97 = mul_98 = None
        getitem_79 = k_embed_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_103 = getitem_79.expand(1, 2, 8, 2, 128)
        getitem_79 = None
        key_20 = hidden_states_103.reshape(1, 16, 2, 128)
        hidden_states_103 = None
        getitem_80 = value_states_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_104 = getitem_80.expand(1, 2, 8, 2, 128)
        getitem_80 = None
        value_20 = hidden_states_104.reshape(1, 16, 2, 128)
        hidden_states_104 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_10 = q_embed_10.contiguous()
        q_embed_10 = None
        key_21 = key_20.contiguous()
        key_20 = None
        value_21 = value_20.contiguous()
        value_20 = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_21,
            value_21,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_10 = key_21 = value_21 = attention_mask_11 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_32 = attn_output_41.reshape(1, 2, -1)
        attn_output_41 = None
        attn_output_42 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_105 = hidden_states_99 + attn_output_43
        hidden_states_99 = attn_output_43 = None
        hidden_states_106 = hidden_states_105.to(torch.float32)
        pow_22 = hidden_states_106.pow(2)
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_64 = variance_21 + 1e-06
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_107 = hidden_states_106 * rsqrt_21
        hidden_states_106 = rsqrt_21 = None
        to_47 = hidden_states_107.to(torch.bfloat16)
        hidden_states_107 = None
        hidden_states_108 = (
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_47
        )
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_47
        ) = None
        linear_74 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_108 = l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_101 = silu_10 * linear_75
        silu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_101,
            l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_101 = l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_109 = hidden_states_105 + down_proj_10
        hidden_states_105 = down_proj_10 = None
        hidden_states_110 = hidden_states_109.to(torch.float32)
        pow_23 = hidden_states_110.pow(2)
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_66 = variance_22 + 1e-06
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_111 = hidden_states_110 * rsqrt_22
        hidden_states_110 = rsqrt_22 = None
        to_49 = hidden_states_111.to(torch.bfloat16)
        hidden_states_111 = None
        hidden_states_112 = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_49
        ) = None
        linear_77 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_34 = linear_77.view((1, 2, -1, 128))
        linear_77 = None
        query_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_35 = linear_78.view((1, 2, -1, 128))
        linear_78 = None
        key_states_11 = view_35.transpose(1, 2)
        view_35 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_36 = linear_79.view((1, 2, -1, 128))
        linear_79 = None
        value_states_11 = view_36.transpose(1, 2)
        view_36 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_104 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 64, None))]
        x2_22 = query_states_11[(Ellipsis, slice(64, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_105 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_104 + mul_105
        mul_104 = mul_105 = None
        mul_106 = key_states_11 * cos_14
        cos_14 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 64, None))]
        x2_23 = key_states_11[(Ellipsis, slice(64, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_107 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_106 + mul_107
        mul_106 = mul_107 = None
        getitem_86 = k_embed_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_113 = getitem_86.expand(1, 2, 8, 2, 128)
        getitem_86 = None
        key_22 = hidden_states_113.reshape(1, 16, 2, 128)
        hidden_states_113 = None
        getitem_87 = value_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_114 = getitem_87.expand(1, 2, 8, 2, 128)
        getitem_87 = None
        value_22 = hidden_states_114.reshape(1, 16, 2, 128)
        hidden_states_114 = None
        attention_mask_12 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_11 = q_embed_11.contiguous()
        q_embed_11 = None
        key_23 = key_22.contiguous()
        key_22 = None
        value_23 = value_22.contiguous()
        value_22 = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_23,
            value_23,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_11 = key_23 = value_23 = attention_mask_12 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_35 = attn_output_45.reshape(1, 2, -1)
        attn_output_45 = None
        attn_output_46 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_115 = hidden_states_109 + attn_output_47
        hidden_states_109 = attn_output_47 = None
        hidden_states_116 = hidden_states_115.to(torch.float32)
        pow_24 = hidden_states_116.pow(2)
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_70 = variance_23 + 1e-06
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_117 = hidden_states_116 * rsqrt_23
        hidden_states_116 = rsqrt_23 = None
        to_51 = hidden_states_117.to(torch.bfloat16)
        hidden_states_117 = None
        hidden_states_118 = (
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_51
        ) = None
        linear_81 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_81, inplace=False)
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_118 = l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_110 = silu_11 * linear_82
        silu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_110,
            l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_110 = l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_119 = hidden_states_115 + down_proj_11
        hidden_states_115 = down_proj_11 = None
        hidden_states_120 = hidden_states_119.to(torch.float32)
        pow_25 = hidden_states_120.pow(2)
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_72 = variance_24 + 1e-06
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_121 = hidden_states_120 * rsqrt_24
        hidden_states_120 = rsqrt_24 = None
        to_53 = hidden_states_121.to(torch.bfloat16)
        hidden_states_121 = None
        hidden_states_122 = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
            * to_53
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            to_53
        ) = None
        linear_84 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_37 = linear_84.view((1, 2, -1, 128))
        linear_84 = None
        query_states_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_38 = linear_85.view((1, 2, -1, 128))
        linear_85 = None
        key_states_12 = view_38.transpose(1, 2)
        view_38 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_39 = linear_86.view((1, 2, -1, 128))
        linear_86 = None
        value_states_12 = view_39.transpose(1, 2)
        view_39 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_113 = query_states_12 * cos_15
        x1_24 = query_states_12[(Ellipsis, slice(None, 64, None))]
        x2_24 = query_states_12[(Ellipsis, slice(64, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_114 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_113 + mul_114
        mul_113 = mul_114 = None
        mul_115 = key_states_12 * cos_15
        cos_15 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 64, None))]
        x2_25 = key_states_12[(Ellipsis, slice(64, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_116 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_115 + mul_116
        mul_115 = mul_116 = None
        getitem_93 = k_embed_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_123 = getitem_93.expand(1, 2, 8, 2, 128)
        getitem_93 = None
        key_24 = hidden_states_123.reshape(1, 16, 2, 128)
        hidden_states_123 = None
        getitem_94 = value_states_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_124 = getitem_94.expand(1, 2, 8, 2, 128)
        getitem_94 = None
        value_24 = hidden_states_124.reshape(1, 16, 2, 128)
        hidden_states_124 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_12 = q_embed_12.contiguous()
        q_embed_12 = None
        key_25 = key_24.contiguous()
        key_24 = None
        value_25 = value_24.contiguous()
        value_24 = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_25,
            value_25,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_12 = key_25 = value_25 = attention_mask_13 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_38 = attn_output_49.reshape(1, 2, -1)
        attn_output_49 = None
        attn_output_50 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_125 = hidden_states_119 + attn_output_51
        hidden_states_119 = attn_output_51 = None
        hidden_states_126 = hidden_states_125.to(torch.float32)
        pow_26 = hidden_states_126.pow(2)
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_76 = variance_25 + 1e-06
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_76)
        add_76 = None
        hidden_states_127 = hidden_states_126 * rsqrt_25
        hidden_states_126 = rsqrt_25 = None
        to_55 = hidden_states_127.to(torch.bfloat16)
        hidden_states_127 = None
        hidden_states_128 = (
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_55
        )
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_55
        ) = None
        linear_88 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_88, inplace=False)
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_128 = l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_119 = silu_12 * linear_89
        silu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_119,
            l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_119 = l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_129 = hidden_states_125 + down_proj_12
        hidden_states_125 = down_proj_12 = None
        hidden_states_130 = hidden_states_129.to(torch.float32)
        pow_27 = hidden_states_130.pow(2)
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_78 = variance_26 + 1e-06
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_131 = hidden_states_130 * rsqrt_26
        hidden_states_130 = rsqrt_26 = None
        to_57 = hidden_states_131.to(torch.bfloat16)
        hidden_states_131 = None
        hidden_states_132 = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
            * to_57
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            to_57
        ) = None
        linear_91 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_40 = linear_91.view((1, 2, -1, 128))
        linear_91 = None
        query_states_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_41 = linear_92.view((1, 2, -1, 128))
        linear_92 = None
        key_states_13 = view_41.transpose(1, 2)
        view_41 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_42 = linear_93.view((1, 2, -1, 128))
        linear_93 = None
        value_states_13 = view_42.transpose(1, 2)
        view_42 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_122 = query_states_13 * cos_16
        x1_26 = query_states_13[(Ellipsis, slice(None, 64, None))]
        x2_26 = query_states_13[(Ellipsis, slice(64, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_123 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_122 + mul_123
        mul_122 = mul_123 = None
        mul_124 = key_states_13 * cos_16
        cos_16 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 64, None))]
        x2_27 = key_states_13[(Ellipsis, slice(64, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_125 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_124 + mul_125
        mul_124 = mul_125 = None
        getitem_100 = k_embed_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_133 = getitem_100.expand(1, 2, 8, 2, 128)
        getitem_100 = None
        key_26 = hidden_states_133.reshape(1, 16, 2, 128)
        hidden_states_133 = None
        getitem_101 = value_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_134 = getitem_101.expand(1, 2, 8, 2, 128)
        getitem_101 = None
        value_26 = hidden_states_134.reshape(1, 16, 2, 128)
        hidden_states_134 = None
        attention_mask_14 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_13 = q_embed_13.contiguous()
        q_embed_13 = None
        key_27 = key_26.contiguous()
        key_26 = None
        value_27 = value_26.contiguous()
        value_26 = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_27,
            value_27,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_13 = key_27 = value_27 = attention_mask_14 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_41 = attn_output_53.reshape(1, 2, -1)
        attn_output_53 = None
        attn_output_54 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_135 = hidden_states_129 + attn_output_55
        hidden_states_129 = attn_output_55 = None
        hidden_states_136 = hidden_states_135.to(torch.float32)
        pow_28 = hidden_states_136.pow(2)
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_82 = variance_27 + 1e-06
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_137 = hidden_states_136 * rsqrt_27
        hidden_states_136 = rsqrt_27 = None
        to_59 = hidden_states_137.to(torch.bfloat16)
        hidden_states_137 = None
        hidden_states_138 = (
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_59
        )
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_59
        ) = None
        linear_95 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_95, inplace=False)
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_138 = l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_128 = silu_13 * linear_96
        silu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_128,
            l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_128 = l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_139 = hidden_states_135 + down_proj_13
        hidden_states_135 = down_proj_13 = None
        hidden_states_140 = hidden_states_139.to(torch.float32)
        pow_29 = hidden_states_140.pow(2)
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_84 = variance_28 + 1e-06
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_84)
        add_84 = None
        hidden_states_141 = hidden_states_140 * rsqrt_28
        hidden_states_140 = rsqrt_28 = None
        to_61 = hidden_states_141.to(torch.bfloat16)
        hidden_states_141 = None
        hidden_states_142 = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
            * to_61
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            to_61
        ) = None
        linear_98 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_43 = linear_98.view((1, 2, -1, 128))
        linear_98 = None
        query_states_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_44 = linear_99.view((1, 2, -1, 128))
        linear_99 = None
        key_states_14 = view_44.transpose(1, 2)
        view_44 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_45 = linear_100.view((1, 2, -1, 128))
        linear_100 = None
        value_states_14 = view_45.transpose(1, 2)
        view_45 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_131 = query_states_14 * cos_17
        x1_28 = query_states_14[(Ellipsis, slice(None, 64, None))]
        x2_28 = query_states_14[(Ellipsis, slice(64, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_132 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_131 + mul_132
        mul_131 = mul_132 = None
        mul_133 = key_states_14 * cos_17
        cos_17 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 64, None))]
        x2_29 = key_states_14[(Ellipsis, slice(64, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_134 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_133 + mul_134
        mul_133 = mul_134 = None
        getitem_107 = k_embed_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_143 = getitem_107.expand(1, 2, 8, 2, 128)
        getitem_107 = None
        key_28 = hidden_states_143.reshape(1, 16, 2, 128)
        hidden_states_143 = None
        getitem_108 = value_states_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_144 = getitem_108.expand(1, 2, 8, 2, 128)
        getitem_108 = None
        value_28 = hidden_states_144.reshape(1, 16, 2, 128)
        hidden_states_144 = None
        attention_mask_15 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_14 = q_embed_14.contiguous()
        q_embed_14 = None
        key_29 = key_28.contiguous()
        key_28 = None
        value_29 = value_28.contiguous()
        value_28 = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_29,
            value_29,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_14 = key_29 = value_29 = attention_mask_15 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_44 = attn_output_57.reshape(1, 2, -1)
        attn_output_57 = None
        attn_output_58 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_145 = hidden_states_139 + attn_output_59
        hidden_states_139 = attn_output_59 = None
        hidden_states_146 = hidden_states_145.to(torch.float32)
        pow_30 = hidden_states_146.pow(2)
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_88 = variance_29 + 1e-06
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_147 = hidden_states_146 * rsqrt_29
        hidden_states_146 = rsqrt_29 = None
        to_63 = hidden_states_147.to(torch.bfloat16)
        hidden_states_147 = None
        hidden_states_148 = (
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_63
        )
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_63
        ) = None
        linear_102 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_102, inplace=False)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_148 = l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_137 = silu_14 * linear_103
        silu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_137,
            l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_137 = l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_149 = hidden_states_145 + down_proj_14
        hidden_states_145 = down_proj_14 = None
        hidden_states_150 = hidden_states_149.to(torch.float32)
        pow_31 = hidden_states_150.pow(2)
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_90 = variance_30 + 1e-06
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_90)
        add_90 = None
        hidden_states_151 = hidden_states_150 * rsqrt_30
        hidden_states_150 = rsqrt_30 = None
        to_65 = hidden_states_151.to(torch.bfloat16)
        hidden_states_151 = None
        hidden_states_152 = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
            * to_65
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            to_65
        ) = None
        linear_105 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_46 = linear_105.view((1, 2, -1, 128))
        linear_105 = None
        query_states_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_47 = linear_106.view((1, 2, -1, 128))
        linear_106 = None
        key_states_15 = view_47.transpose(1, 2)
        view_47 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_48 = linear_107.view((1, 2, -1, 128))
        linear_107 = None
        value_states_15 = view_48.transpose(1, 2)
        view_48 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_140 = query_states_15 * cos_18
        x1_30 = query_states_15[(Ellipsis, slice(None, 64, None))]
        x2_30 = query_states_15[(Ellipsis, slice(64, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_141 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_140 + mul_141
        mul_140 = mul_141 = None
        mul_142 = key_states_15 * cos_18
        cos_18 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 64, None))]
        x2_31 = key_states_15[(Ellipsis, slice(64, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_143 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_142 + mul_143
        mul_142 = mul_143 = None
        getitem_114 = k_embed_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_153 = getitem_114.expand(1, 2, 8, 2, 128)
        getitem_114 = None
        key_30 = hidden_states_153.reshape(1, 16, 2, 128)
        hidden_states_153 = None
        getitem_115 = value_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_154 = getitem_115.expand(1, 2, 8, 2, 128)
        getitem_115 = None
        value_30 = hidden_states_154.reshape(1, 16, 2, 128)
        hidden_states_154 = None
        attention_mask_16 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_15 = q_embed_15.contiguous()
        q_embed_15 = None
        key_31 = key_30.contiguous()
        key_30 = None
        value_31 = value_30.contiguous()
        value_30 = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_31,
            value_31,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_15 = key_31 = value_31 = attention_mask_16 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_47 = attn_output_61.reshape(1, 2, -1)
        attn_output_61 = None
        attn_output_62 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_155 = hidden_states_149 + attn_output_63
        hidden_states_149 = attn_output_63 = None
        hidden_states_156 = hidden_states_155.to(torch.float32)
        pow_32 = hidden_states_156.pow(2)
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_94 = variance_31 + 1e-06
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_157 = hidden_states_156 * rsqrt_31
        hidden_states_156 = rsqrt_31 = None
        to_67 = hidden_states_157.to(torch.bfloat16)
        hidden_states_157 = None
        hidden_states_158 = (
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_67
        )
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_67
        ) = None
        linear_109 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_109, inplace=False)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_158 = l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_146 = silu_15 * linear_110
        silu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_146,
            l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_146 = l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_159 = hidden_states_155 + down_proj_15
        hidden_states_155 = down_proj_15 = None
        hidden_states_160 = hidden_states_159.to(torch.float32)
        pow_33 = hidden_states_160.pow(2)
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_96 = variance_32 + 1e-06
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_161 = hidden_states_160 * rsqrt_32
        hidden_states_160 = rsqrt_32 = None
        to_69 = hidden_states_161.to(torch.bfloat16)
        hidden_states_161 = None
        hidden_states_162 = (
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
            * to_69
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            to_69
        ) = None
        linear_112 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_49 = linear_112.view((1, 2, -1, 128))
        linear_112 = None
        query_states_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_50 = linear_113.view((1, 2, -1, 128))
        linear_113 = None
        key_states_16 = view_50.transpose(1, 2)
        view_50 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_162 = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_51 = linear_114.view((1, 2, -1, 128))
        linear_114 = None
        value_states_16 = view_51.transpose(1, 2)
        view_51 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_149 = query_states_16 * cos_19
        x1_32 = query_states_16[(Ellipsis, slice(None, 64, None))]
        x2_32 = query_states_16[(Ellipsis, slice(64, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_150 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_149 + mul_150
        mul_149 = mul_150 = None
        mul_151 = key_states_16 * cos_19
        cos_19 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 64, None))]
        x2_33 = key_states_16[(Ellipsis, slice(64, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_152 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_151 + mul_152
        mul_151 = mul_152 = None
        getitem_121 = k_embed_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_163 = getitem_121.expand(1, 2, 8, 2, 128)
        getitem_121 = None
        key_32 = hidden_states_163.reshape(1, 16, 2, 128)
        hidden_states_163 = None
        getitem_122 = value_states_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_164 = getitem_122.expand(1, 2, 8, 2, 128)
        getitem_122 = None
        value_32 = hidden_states_164.reshape(1, 16, 2, 128)
        hidden_states_164 = None
        attention_mask_17 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_16 = q_embed_16.contiguous()
        q_embed_16 = None
        key_33 = key_32.contiguous()
        key_32 = None
        value_33 = value_32.contiguous()
        value_32 = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_33,
            value_33,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_16 = key_33 = value_33 = attention_mask_17 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_50 = attn_output_65.reshape(1, 2, -1)
        attn_output_65 = None
        attn_output_66 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_165 = hidden_states_159 + attn_output_67
        hidden_states_159 = attn_output_67 = None
        hidden_states_166 = hidden_states_165.to(torch.float32)
        pow_34 = hidden_states_166.pow(2)
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_100 = variance_33 + 1e-06
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_100)
        add_100 = None
        hidden_states_167 = hidden_states_166 * rsqrt_33
        hidden_states_166 = rsqrt_33 = None
        to_71 = hidden_states_167.to(torch.bfloat16)
        hidden_states_167 = None
        hidden_states_168 = (
            l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_71
        )
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_71
        ) = None
        linear_116 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_116, inplace=False)
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_168 = l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_155 = silu_16 * linear_117
        silu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_155,
            l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_155 = l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_169 = hidden_states_165 + down_proj_16
        hidden_states_165 = down_proj_16 = None
        hidden_states_170 = hidden_states_169.to(torch.float32)
        pow_35 = hidden_states_170.pow(2)
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_102 = variance_34 + 1e-06
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_171 = hidden_states_170 * rsqrt_34
        hidden_states_170 = rsqrt_34 = None
        to_73 = hidden_states_171.to(torch.bfloat16)
        hidden_states_171 = None
        hidden_states_172 = (
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
            * to_73
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            to_73
        ) = None
        linear_119 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_52 = linear_119.view((1, 2, -1, 128))
        linear_119 = None
        query_states_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_53 = linear_120.view((1, 2, -1, 128))
        linear_120 = None
        key_states_17 = view_53.transpose(1, 2)
        view_53 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_54 = linear_121.view((1, 2, -1, 128))
        linear_121 = None
        value_states_17 = view_54.transpose(1, 2)
        view_54 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_158 = query_states_17 * cos_20
        x1_34 = query_states_17[(Ellipsis, slice(None, 64, None))]
        x2_34 = query_states_17[(Ellipsis, slice(64, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_159 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_158 + mul_159
        mul_158 = mul_159 = None
        mul_160 = key_states_17 * cos_20
        cos_20 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 64, None))]
        x2_35 = key_states_17[(Ellipsis, slice(64, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_161 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_160 + mul_161
        mul_160 = mul_161 = None
        getitem_128 = k_embed_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_173 = getitem_128.expand(1, 2, 8, 2, 128)
        getitem_128 = None
        key_34 = hidden_states_173.reshape(1, 16, 2, 128)
        hidden_states_173 = None
        getitem_129 = value_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_174 = getitem_129.expand(1, 2, 8, 2, 128)
        getitem_129 = None
        value_34 = hidden_states_174.reshape(1, 16, 2, 128)
        hidden_states_174 = None
        attention_mask_18 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_17 = q_embed_17.contiguous()
        q_embed_17 = None
        key_35 = key_34.contiguous()
        key_34 = None
        value_35 = value_34.contiguous()
        value_34 = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_35,
            value_35,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_17 = key_35 = value_35 = attention_mask_18 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_53 = attn_output_69.reshape(1, 2, -1)
        attn_output_69 = None
        attn_output_70 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_175 = hidden_states_169 + attn_output_71
        hidden_states_169 = attn_output_71 = None
        hidden_states_176 = hidden_states_175.to(torch.float32)
        pow_36 = hidden_states_176.pow(2)
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_106 = variance_35 + 1e-06
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_177 = hidden_states_176 * rsqrt_35
        hidden_states_176 = rsqrt_35 = None
        to_75 = hidden_states_177.to(torch.bfloat16)
        hidden_states_177 = None
        hidden_states_178 = (
            l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_75
        )
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_75
        ) = None
        linear_123 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_123, inplace=False)
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_178 = l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_164 = silu_17 * linear_124
        silu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_164,
            l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_164 = l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_179 = hidden_states_175 + down_proj_17
        hidden_states_175 = down_proj_17 = None
        hidden_states_180 = hidden_states_179.to(torch.float32)
        pow_37 = hidden_states_180.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_108 = variance_36 + 1e-06
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_181 = hidden_states_180 * rsqrt_36
        hidden_states_180 = rsqrt_36 = None
        to_77 = hidden_states_181.to(torch.bfloat16)
        hidden_states_181 = None
        hidden_states_182 = (
            l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
            * to_77
        )
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            to_77
        ) = None
        linear_126 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_55 = linear_126.view((1, 2, -1, 128))
        linear_126 = None
        query_states_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_56 = linear_127.view((1, 2, -1, 128))
        linear_127 = None
        key_states_18 = view_56.transpose(1, 2)
        view_56 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_57 = linear_128.view((1, 2, -1, 128))
        linear_128 = None
        value_states_18 = view_57.transpose(1, 2)
        view_57 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_167 = query_states_18 * cos_21
        x1_36 = query_states_18[(Ellipsis, slice(None, 64, None))]
        x2_36 = query_states_18[(Ellipsis, slice(64, None, None))]
        query_states_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_37 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_168 = cat_37 * sin_21
        cat_37 = None
        q_embed_18 = mul_167 + mul_168
        mul_167 = mul_168 = None
        mul_169 = key_states_18 * cos_21
        cos_21 = None
        x1_37 = key_states_18[(Ellipsis, slice(None, 64, None))]
        x2_37 = key_states_18[(Ellipsis, slice(64, None, None))]
        key_states_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_38 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_170 = cat_38 * sin_21
        cat_38 = sin_21 = None
        k_embed_18 = mul_169 + mul_170
        mul_169 = mul_170 = None
        getitem_135 = k_embed_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_183 = getitem_135.expand(1, 2, 8, 2, 128)
        getitem_135 = None
        key_36 = hidden_states_183.reshape(1, 16, 2, 128)
        hidden_states_183 = None
        getitem_136 = value_states_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_184 = getitem_136.expand(1, 2, 8, 2, 128)
        getitem_136 = None
        value_36 = hidden_states_184.reshape(1, 16, 2, 128)
        hidden_states_184 = None
        attention_mask_19 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_18 = q_embed_18.contiguous()
        q_embed_18 = None
        key_37 = key_36.contiguous()
        key_36 = None
        value_37 = value_36.contiguous()
        value_36 = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_37,
            value_37,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_18 = key_37 = value_37 = attention_mask_19 = None
        transpose_76 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_76.contiguous()
        transpose_76 = None
        reshape_56 = attn_output_73.reshape(1, 2, -1)
        attn_output_73 = None
        attn_output_74 = reshape_56.contiguous()
        reshape_56 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_185 = hidden_states_179 + attn_output_75
        hidden_states_179 = attn_output_75 = None
        hidden_states_186 = hidden_states_185.to(torch.float32)
        pow_38 = hidden_states_186.pow(2)
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_112 = variance_37 + 1e-06
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_187 = hidden_states_186 * rsqrt_37
        hidden_states_186 = rsqrt_37 = None
        to_79 = hidden_states_187.to(torch.bfloat16)
        hidden_states_187 = None
        hidden_states_188 = (
            l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
            * to_79
        )
        l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            to_79
        ) = None
        linear_130 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_130, inplace=False)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_188 = l_self_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_173 = silu_18 * linear_131
        silu_18 = linear_131 = None
        down_proj_18 = torch._C._nn.linear(
            mul_173,
            l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_173 = l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_189 = hidden_states_185 + down_proj_18
        hidden_states_185 = down_proj_18 = None
        hidden_states_190 = hidden_states_189.to(torch.float32)
        pow_39 = hidden_states_190.pow(2)
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_114 = variance_38 + 1e-06
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_114)
        add_114 = None
        hidden_states_191 = hidden_states_190 * rsqrt_38
        hidden_states_190 = rsqrt_38 = None
        to_81 = hidden_states_191.to(torch.bfloat16)
        hidden_states_191 = None
        hidden_states_192 = (
            l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
            * to_81
        )
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            to_81
        ) = None
        linear_133 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_58 = linear_133.view((1, 2, -1, 128))
        linear_133 = None
        query_states_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_59 = linear_134.view((1, 2, -1, 128))
        linear_134 = None
        key_states_19 = view_59.transpose(1, 2)
        view_59 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_192 = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_60 = linear_135.view((1, 2, -1, 128))
        linear_135 = None
        value_states_19 = view_60.transpose(1, 2)
        view_60 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_176 = query_states_19 * cos_22
        x1_38 = query_states_19[(Ellipsis, slice(None, 64, None))]
        x2_38 = query_states_19[(Ellipsis, slice(64, None, None))]
        query_states_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_39 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_177 = cat_39 * sin_22
        cat_39 = None
        q_embed_19 = mul_176 + mul_177
        mul_176 = mul_177 = None
        mul_178 = key_states_19 * cos_22
        cos_22 = None
        x1_39 = key_states_19[(Ellipsis, slice(None, 64, None))]
        x2_39 = key_states_19[(Ellipsis, slice(64, None, None))]
        key_states_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_40 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_179 = cat_40 * sin_22
        cat_40 = sin_22 = None
        k_embed_19 = mul_178 + mul_179
        mul_178 = mul_179 = None
        getitem_142 = k_embed_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_193 = getitem_142.expand(1, 2, 8, 2, 128)
        getitem_142 = None
        key_38 = hidden_states_193.reshape(1, 16, 2, 128)
        hidden_states_193 = None
        getitem_143 = value_states_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_194 = getitem_143.expand(1, 2, 8, 2, 128)
        getitem_143 = None
        value_38 = hidden_states_194.reshape(1, 16, 2, 128)
        hidden_states_194 = None
        attention_mask_20 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_19 = q_embed_19.contiguous()
        q_embed_19 = None
        key_39 = key_38.contiguous()
        key_38 = None
        value_39 = value_38.contiguous()
        value_38 = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_39,
            value_39,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_19 = key_39 = value_39 = attention_mask_20 = None
        transpose_80 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_80.contiguous()
        transpose_80 = None
        reshape_59 = attn_output_77.reshape(1, 2, -1)
        attn_output_77 = None
        attn_output_78 = reshape_59.contiguous()
        reshape_59 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_195 = hidden_states_189 + attn_output_79
        hidden_states_189 = attn_output_79 = None
        hidden_states_196 = hidden_states_195.to(torch.float32)
        pow_40 = hidden_states_196.pow(2)
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_118 = variance_39 + 1e-06
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_197 = hidden_states_196 * rsqrt_39
        hidden_states_196 = rsqrt_39 = None
        to_83 = hidden_states_197.to(torch.bfloat16)
        hidden_states_197 = None
        hidden_states_198 = (
            l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
            * to_83
        )
        l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            to_83
        ) = None
        linear_137 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_137, inplace=False)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_198 = l_self_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_182 = silu_19 * linear_138
        silu_19 = linear_138 = None
        down_proj_19 = torch._C._nn.linear(
            mul_182,
            l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_182 = l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_199 = hidden_states_195 + down_proj_19
        hidden_states_195 = down_proj_19 = None
        hidden_states_200 = hidden_states_199.to(torch.float32)
        pow_41 = hidden_states_200.pow(2)
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_120 = variance_40 + 1e-06
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_201 = hidden_states_200 * rsqrt_40
        hidden_states_200 = rsqrt_40 = None
        to_85 = hidden_states_201.to(torch.bfloat16)
        hidden_states_201 = None
        hidden_states_202 = (
            l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
            * to_85
        )
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            to_85
        ) = None
        linear_140 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_61 = linear_140.view((1, 2, -1, 128))
        linear_140 = None
        query_states_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_62 = linear_141.view((1, 2, -1, 128))
        linear_141 = None
        key_states_20 = view_62.transpose(1, 2)
        view_62 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_202 = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_63 = linear_142.view((1, 2, -1, 128))
        linear_142 = None
        value_states_20 = view_63.transpose(1, 2)
        view_63 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_185 = query_states_20 * cos_23
        x1_40 = query_states_20[(Ellipsis, slice(None, 64, None))]
        x2_40 = query_states_20[(Ellipsis, slice(64, None, None))]
        query_states_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_41 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_186 = cat_41 * sin_23
        cat_41 = None
        q_embed_20 = mul_185 + mul_186
        mul_185 = mul_186 = None
        mul_187 = key_states_20 * cos_23
        cos_23 = None
        x1_41 = key_states_20[(Ellipsis, slice(None, 64, None))]
        x2_41 = key_states_20[(Ellipsis, slice(64, None, None))]
        key_states_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_42 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_188 = cat_42 * sin_23
        cat_42 = sin_23 = None
        k_embed_20 = mul_187 + mul_188
        mul_187 = mul_188 = None
        getitem_149 = k_embed_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_203 = getitem_149.expand(1, 2, 8, 2, 128)
        getitem_149 = None
        key_40 = hidden_states_203.reshape(1, 16, 2, 128)
        hidden_states_203 = None
        getitem_150 = value_states_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_204 = getitem_150.expand(1, 2, 8, 2, 128)
        getitem_150 = None
        value_40 = hidden_states_204.reshape(1, 16, 2, 128)
        hidden_states_204 = None
        attention_mask_21 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_20 = q_embed_20.contiguous()
        q_embed_20 = None
        key_41 = key_40.contiguous()
        key_40 = None
        value_41 = value_40.contiguous()
        value_40 = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_41,
            value_41,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_20 = key_41 = value_41 = attention_mask_21 = None
        transpose_84 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_84.contiguous()
        transpose_84 = None
        reshape_62 = attn_output_81.reshape(1, 2, -1)
        attn_output_81 = None
        attn_output_82 = reshape_62.contiguous()
        reshape_62 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_205 = hidden_states_199 + attn_output_83
        hidden_states_199 = attn_output_83 = None
        hidden_states_206 = hidden_states_205.to(torch.float32)
        pow_42 = hidden_states_206.pow(2)
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_124 = variance_41 + 1e-06
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_207 = hidden_states_206 * rsqrt_41
        hidden_states_206 = rsqrt_41 = None
        to_87 = hidden_states_207.to(torch.bfloat16)
        hidden_states_207 = None
        hidden_states_208 = (
            l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
            * to_87
        )
        l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            to_87
        ) = None
        linear_144 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_144, inplace=False)
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_208 = l_self_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_191 = silu_20 * linear_145
        silu_20 = linear_145 = None
        down_proj_20 = torch._C._nn.linear(
            mul_191,
            l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_191 = l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_209 = hidden_states_205 + down_proj_20
        hidden_states_205 = down_proj_20 = None
        hidden_states_210 = hidden_states_209.to(torch.float32)
        pow_43 = hidden_states_210.pow(2)
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_126 = variance_42 + 1e-06
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_211 = hidden_states_210 * rsqrt_42
        hidden_states_210 = rsqrt_42 = None
        to_89 = hidden_states_211.to(torch.bfloat16)
        hidden_states_211 = None
        hidden_states_212 = (
            l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
            * to_89
        )
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            to_89
        ) = None
        linear_147 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_64 = linear_147.view((1, 2, -1, 128))
        linear_147 = None
        query_states_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_65 = linear_148.view((1, 2, -1, 128))
        linear_148 = None
        key_states_21 = view_65.transpose(1, 2)
        view_65 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_66 = linear_149.view((1, 2, -1, 128))
        linear_149 = None
        value_states_21 = view_66.transpose(1, 2)
        view_66 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_194 = query_states_21 * cos_24
        x1_42 = query_states_21[(Ellipsis, slice(None, 64, None))]
        x2_42 = query_states_21[(Ellipsis, slice(64, None, None))]
        query_states_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_43 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_195 = cat_43 * sin_24
        cat_43 = None
        q_embed_21 = mul_194 + mul_195
        mul_194 = mul_195 = None
        mul_196 = key_states_21 * cos_24
        cos_24 = None
        x1_43 = key_states_21[(Ellipsis, slice(None, 64, None))]
        x2_43 = key_states_21[(Ellipsis, slice(64, None, None))]
        key_states_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_44 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_197 = cat_44 * sin_24
        cat_44 = sin_24 = None
        k_embed_21 = mul_196 + mul_197
        mul_196 = mul_197 = None
        getitem_156 = k_embed_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_213 = getitem_156.expand(1, 2, 8, 2, 128)
        getitem_156 = None
        key_42 = hidden_states_213.reshape(1, 16, 2, 128)
        hidden_states_213 = None
        getitem_157 = value_states_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_214 = getitem_157.expand(1, 2, 8, 2, 128)
        getitem_157 = None
        value_42 = hidden_states_214.reshape(1, 16, 2, 128)
        hidden_states_214 = None
        attention_mask_22 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_21 = q_embed_21.contiguous()
        q_embed_21 = None
        key_43 = key_42.contiguous()
        key_42 = None
        value_43 = value_42.contiguous()
        value_42 = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_43,
            value_43,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_21 = key_43 = value_43 = attention_mask_22 = None
        transpose_88 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_88.contiguous()
        transpose_88 = None
        reshape_65 = attn_output_85.reshape(1, 2, -1)
        attn_output_85 = None
        attn_output_86 = reshape_65.contiguous()
        reshape_65 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_215 = hidden_states_209 + attn_output_87
        hidden_states_209 = attn_output_87 = None
        hidden_states_216 = hidden_states_215.to(torch.float32)
        pow_44 = hidden_states_216.pow(2)
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_130 = variance_43 + 1e-06
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_130)
        add_130 = None
        hidden_states_217 = hidden_states_216 * rsqrt_43
        hidden_states_216 = rsqrt_43 = None
        to_91 = hidden_states_217.to(torch.bfloat16)
        hidden_states_217 = None
        hidden_states_218 = (
            l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
            * to_91
        )
        l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            to_91
        ) = None
        linear_151 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_151, inplace=False)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_218 = l_self_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_200 = silu_21 * linear_152
        silu_21 = linear_152 = None
        down_proj_21 = torch._C._nn.linear(
            mul_200,
            l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_200 = l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_219 = hidden_states_215 + down_proj_21
        hidden_states_215 = down_proj_21 = None
        hidden_states_220 = hidden_states_219.to(torch.float32)
        pow_45 = hidden_states_220.pow(2)
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_132 = variance_44 + 1e-06
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_132)
        add_132 = None
        hidden_states_221 = hidden_states_220 * rsqrt_44
        hidden_states_220 = rsqrt_44 = None
        to_93 = hidden_states_221.to(torch.bfloat16)
        hidden_states_221 = None
        hidden_states_222 = (
            l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
            * to_93
        )
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            to_93
        ) = None
        linear_154 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_67 = linear_154.view((1, 2, -1, 128))
        linear_154 = None
        query_states_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_68 = linear_155.view((1, 2, -1, 128))
        linear_155 = None
        key_states_22 = view_68.transpose(1, 2)
        view_68 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_222 = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_69 = linear_156.view((1, 2, -1, 128))
        linear_156 = None
        value_states_22 = view_69.transpose(1, 2)
        view_69 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_203 = query_states_22 * cos_25
        x1_44 = query_states_22[(Ellipsis, slice(None, 64, None))]
        x2_44 = query_states_22[(Ellipsis, slice(64, None, None))]
        query_states_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_45 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_204 = cat_45 * sin_25
        cat_45 = None
        q_embed_22 = mul_203 + mul_204
        mul_203 = mul_204 = None
        mul_205 = key_states_22 * cos_25
        cos_25 = None
        x1_45 = key_states_22[(Ellipsis, slice(None, 64, None))]
        x2_45 = key_states_22[(Ellipsis, slice(64, None, None))]
        key_states_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_46 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_206 = cat_46 * sin_25
        cat_46 = sin_25 = None
        k_embed_22 = mul_205 + mul_206
        mul_205 = mul_206 = None
        getitem_163 = k_embed_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_223 = getitem_163.expand(1, 2, 8, 2, 128)
        getitem_163 = None
        key_44 = hidden_states_223.reshape(1, 16, 2, 128)
        hidden_states_223 = None
        getitem_164 = value_states_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_224 = getitem_164.expand(1, 2, 8, 2, 128)
        getitem_164 = None
        value_44 = hidden_states_224.reshape(1, 16, 2, 128)
        hidden_states_224 = None
        attention_mask_23 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_22 = q_embed_22.contiguous()
        q_embed_22 = None
        key_45 = key_44.contiguous()
        key_44 = None
        value_45 = value_44.contiguous()
        value_44 = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_45,
            value_45,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_22 = key_45 = value_45 = attention_mask_23 = None
        transpose_92 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_92.contiguous()
        transpose_92 = None
        reshape_68 = attn_output_89.reshape(1, 2, -1)
        attn_output_89 = None
        attn_output_90 = reshape_68.contiguous()
        reshape_68 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_225 = hidden_states_219 + attn_output_91
        hidden_states_219 = attn_output_91 = None
        hidden_states_226 = hidden_states_225.to(torch.float32)
        pow_46 = hidden_states_226.pow(2)
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_136 = variance_45 + 1e-06
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_227 = hidden_states_226 * rsqrt_45
        hidden_states_226 = rsqrt_45 = None
        to_95 = hidden_states_227.to(torch.bfloat16)
        hidden_states_227 = None
        hidden_states_228 = (
            l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
            * to_95
        )
        l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            to_95
        ) = None
        linear_158 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_158, inplace=False)
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_228 = l_self_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_209 = silu_22 * linear_159
        silu_22 = linear_159 = None
        down_proj_22 = torch._C._nn.linear(
            mul_209,
            l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_209 = l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_229 = hidden_states_225 + down_proj_22
        hidden_states_225 = down_proj_22 = None
        hidden_states_230 = hidden_states_229.to(torch.float32)
        pow_47 = hidden_states_230.pow(2)
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_138 = variance_46 + 1e-06
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_231 = hidden_states_230 * rsqrt_46
        hidden_states_230 = rsqrt_46 = None
        to_97 = hidden_states_231.to(torch.bfloat16)
        hidden_states_231 = None
        hidden_states_232 = (
            l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
            * to_97
        )
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            to_97
        ) = None
        linear_161 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_70 = linear_161.view((1, 2, -1, 128))
        linear_161 = None
        query_states_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_71 = linear_162.view((1, 2, -1, 128))
        linear_162 = None
        key_states_23 = view_71.transpose(1, 2)
        view_71 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_72 = linear_163.view((1, 2, -1, 128))
        linear_163 = None
        value_states_23 = view_72.transpose(1, 2)
        view_72 = None
        cos_26 = cos_2.unsqueeze(1)
        sin_26 = sin_2.unsqueeze(1)
        mul_212 = query_states_23 * cos_26
        x1_46 = query_states_23[(Ellipsis, slice(None, 64, None))]
        x2_46 = query_states_23[(Ellipsis, slice(64, None, None))]
        query_states_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_47 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_213 = cat_47 * sin_26
        cat_47 = None
        q_embed_23 = mul_212 + mul_213
        mul_212 = mul_213 = None
        mul_214 = key_states_23 * cos_26
        cos_26 = None
        x1_47 = key_states_23[(Ellipsis, slice(None, 64, None))]
        x2_47 = key_states_23[(Ellipsis, slice(64, None, None))]
        key_states_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_48 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_215 = cat_48 * sin_26
        cat_48 = sin_26 = None
        k_embed_23 = mul_214 + mul_215
        mul_214 = mul_215 = None
        getitem_170 = k_embed_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_233 = getitem_170.expand(1, 2, 8, 2, 128)
        getitem_170 = None
        key_46 = hidden_states_233.reshape(1, 16, 2, 128)
        hidden_states_233 = None
        getitem_171 = value_states_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_234 = getitem_171.expand(1, 2, 8, 2, 128)
        getitem_171 = None
        value_46 = hidden_states_234.reshape(1, 16, 2, 128)
        hidden_states_234 = None
        attention_mask_24 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_23 = q_embed_23.contiguous()
        q_embed_23 = None
        key_47 = key_46.contiguous()
        key_46 = None
        value_47 = value_46.contiguous()
        value_46 = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_47,
            value_47,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_23 = key_47 = value_47 = attention_mask_24 = None
        transpose_96 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_96.contiguous()
        transpose_96 = None
        reshape_71 = attn_output_93.reshape(1, 2, -1)
        attn_output_93 = None
        attn_output_94 = reshape_71.contiguous()
        reshape_71 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_235 = hidden_states_229 + attn_output_95
        hidden_states_229 = attn_output_95 = None
        hidden_states_236 = hidden_states_235.to(torch.float32)
        pow_48 = hidden_states_236.pow(2)
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_142 = variance_47 + 1e-06
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_237 = hidden_states_236 * rsqrt_47
        hidden_states_236 = rsqrt_47 = None
        to_99 = hidden_states_237.to(torch.bfloat16)
        hidden_states_237 = None
        hidden_states_238 = (
            l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
            * to_99
        )
        l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            to_99
        ) = None
        linear_165 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_165, inplace=False)
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_238 = l_self_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_218 = silu_23 * linear_166
        silu_23 = linear_166 = None
        down_proj_23 = torch._C._nn.linear(
            mul_218,
            l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_218 = l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_239 = hidden_states_235 + down_proj_23
        hidden_states_235 = down_proj_23 = None
        hidden_states_240 = hidden_states_239.to(torch.float32)
        pow_49 = hidden_states_240.pow(2)
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_144 = variance_48 + 1e-06
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_241 = hidden_states_240 * rsqrt_48
        hidden_states_240 = rsqrt_48 = None
        to_101 = hidden_states_241.to(torch.bfloat16)
        hidden_states_241 = None
        hidden_states_242 = (
            l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
            * to_101
        )
        l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            to_101
        ) = None
        linear_168 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_73 = linear_168.view((1, 2, -1, 128))
        linear_168 = None
        query_states_24 = view_73.transpose(1, 2)
        view_73 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_74 = linear_169.view((1, 2, -1, 128))
        linear_169 = None
        key_states_24 = view_74.transpose(1, 2)
        view_74 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_242 = l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_75 = linear_170.view((1, 2, -1, 128))
        linear_170 = None
        value_states_24 = view_75.transpose(1, 2)
        view_75 = None
        cos_27 = cos_2.unsqueeze(1)
        sin_27 = sin_2.unsqueeze(1)
        mul_221 = query_states_24 * cos_27
        x1_48 = query_states_24[(Ellipsis, slice(None, 64, None))]
        x2_48 = query_states_24[(Ellipsis, slice(64, None, None))]
        query_states_24 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_49 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_222 = cat_49 * sin_27
        cat_49 = None
        q_embed_24 = mul_221 + mul_222
        mul_221 = mul_222 = None
        mul_223 = key_states_24 * cos_27
        cos_27 = None
        x1_49 = key_states_24[(Ellipsis, slice(None, 64, None))]
        x2_49 = key_states_24[(Ellipsis, slice(64, None, None))]
        key_states_24 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_50 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_224 = cat_50 * sin_27
        cat_50 = sin_27 = None
        k_embed_24 = mul_223 + mul_224
        mul_223 = mul_224 = None
        getitem_177 = k_embed_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_243 = getitem_177.expand(1, 2, 8, 2, 128)
        getitem_177 = None
        key_48 = hidden_states_243.reshape(1, 16, 2, 128)
        hidden_states_243 = None
        getitem_178 = value_states_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_244 = getitem_178.expand(1, 2, 8, 2, 128)
        getitem_178 = None
        value_48 = hidden_states_244.reshape(1, 16, 2, 128)
        hidden_states_244 = None
        attention_mask_25 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_24 = q_embed_24.contiguous()
        q_embed_24 = None
        key_49 = key_48.contiguous()
        key_48 = None
        value_49 = value_48.contiguous()
        value_48 = None
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_49,
            value_49,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_24 = key_49 = value_49 = attention_mask_25 = None
        transpose_100 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_100.contiguous()
        transpose_100 = None
        reshape_74 = attn_output_97.reshape(1, 2, -1)
        attn_output_97 = None
        attn_output_98 = reshape_74.contiguous()
        reshape_74 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_245 = hidden_states_239 + attn_output_99
        hidden_states_239 = attn_output_99 = None
        hidden_states_246 = hidden_states_245.to(torch.float32)
        pow_50 = hidden_states_246.pow(2)
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_148 = variance_49 + 1e-06
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_148)
        add_148 = None
        hidden_states_247 = hidden_states_246 * rsqrt_49
        hidden_states_246 = rsqrt_49 = None
        to_103 = hidden_states_247.to(torch.bfloat16)
        hidden_states_247 = None
        hidden_states_248 = (
            l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
            * to_103
        )
        l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            to_103
        ) = None
        linear_172 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_172, inplace=False)
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_248 = l_self_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_227 = silu_24 * linear_173
        silu_24 = linear_173 = None
        down_proj_24 = torch._C._nn.linear(
            mul_227,
            l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_227 = l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_249 = hidden_states_245 + down_proj_24
        hidden_states_245 = down_proj_24 = None
        hidden_states_250 = hidden_states_249.to(torch.float32)
        pow_51 = hidden_states_250.pow(2)
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_150 = variance_50 + 1e-06
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_251 = hidden_states_250 * rsqrt_50
        hidden_states_250 = rsqrt_50 = None
        to_105 = hidden_states_251.to(torch.bfloat16)
        hidden_states_251 = None
        hidden_states_252 = (
            l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
            * to_105
        )
        l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            to_105
        ) = None
        linear_175 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_76 = linear_175.view((1, 2, -1, 128))
        linear_175 = None
        query_states_25 = view_76.transpose(1, 2)
        view_76 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_77 = linear_176.view((1, 2, -1, 128))
        linear_176 = None
        key_states_25 = view_77.transpose(1, 2)
        view_77 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_252 = l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_78 = linear_177.view((1, 2, -1, 128))
        linear_177 = None
        value_states_25 = view_78.transpose(1, 2)
        view_78 = None
        cos_28 = cos_2.unsqueeze(1)
        sin_28 = sin_2.unsqueeze(1)
        mul_230 = query_states_25 * cos_28
        x1_50 = query_states_25[(Ellipsis, slice(None, 64, None))]
        x2_50 = query_states_25[(Ellipsis, slice(64, None, None))]
        query_states_25 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_51 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_231 = cat_51 * sin_28
        cat_51 = None
        q_embed_25 = mul_230 + mul_231
        mul_230 = mul_231 = None
        mul_232 = key_states_25 * cos_28
        cos_28 = None
        x1_51 = key_states_25[(Ellipsis, slice(None, 64, None))]
        x2_51 = key_states_25[(Ellipsis, slice(64, None, None))]
        key_states_25 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_52 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_233 = cat_52 * sin_28
        cat_52 = sin_28 = None
        k_embed_25 = mul_232 + mul_233
        mul_232 = mul_233 = None
        getitem_184 = k_embed_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_253 = getitem_184.expand(1, 2, 8, 2, 128)
        getitem_184 = None
        key_50 = hidden_states_253.reshape(1, 16, 2, 128)
        hidden_states_253 = None
        getitem_185 = value_states_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_254 = getitem_185.expand(1, 2, 8, 2, 128)
        getitem_185 = None
        value_50 = hidden_states_254.reshape(1, 16, 2, 128)
        hidden_states_254 = None
        attention_mask_26 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_25 = q_embed_25.contiguous()
        q_embed_25 = None
        key_51 = key_50.contiguous()
        key_50 = None
        value_51 = value_50.contiguous()
        value_50 = None
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_51,
            value_51,
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_25 = key_51 = value_51 = attention_mask_26 = None
        transpose_104 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_104.contiguous()
        transpose_104 = None
        reshape_77 = attn_output_101.reshape(1, 2, -1)
        attn_output_101 = None
        attn_output_102 = reshape_77.contiguous()
        reshape_77 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_255 = hidden_states_249 + attn_output_103
        hidden_states_249 = attn_output_103 = None
        hidden_states_256 = hidden_states_255.to(torch.float32)
        pow_52 = hidden_states_256.pow(2)
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_154 = variance_51 + 1e-06
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_154)
        add_154 = None
        hidden_states_257 = hidden_states_256 * rsqrt_51
        hidden_states_256 = rsqrt_51 = None
        to_107 = hidden_states_257.to(torch.bfloat16)
        hidden_states_257 = None
        hidden_states_258 = (
            l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
            * to_107
        )
        l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            to_107
        ) = None
        linear_179 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_179, inplace=False)
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_258 = l_self_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_236 = silu_25 * linear_180
        silu_25 = linear_180 = None
        down_proj_25 = torch._C._nn.linear(
            mul_236,
            l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_236 = l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_259 = hidden_states_255 + down_proj_25
        hidden_states_255 = down_proj_25 = None
        hidden_states_260 = hidden_states_259.to(torch.float32)
        pow_53 = hidden_states_260.pow(2)
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_156 = variance_52 + 1e-06
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_156)
        add_156 = None
        hidden_states_261 = hidden_states_260 * rsqrt_52
        hidden_states_260 = rsqrt_52 = None
        to_109 = hidden_states_261.to(torch.bfloat16)
        hidden_states_261 = None
        hidden_states_262 = (
            l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
            * to_109
        )
        l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            to_109
        ) = None
        linear_182 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_79 = linear_182.view((1, 2, -1, 128))
        linear_182 = None
        query_states_26 = view_79.transpose(1, 2)
        view_79 = None
        linear_183 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_80 = linear_183.view((1, 2, -1, 128))
        linear_183 = None
        key_states_26 = view_80.transpose(1, 2)
        view_80 = None
        linear_184 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_262 = l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_81 = linear_184.view((1, 2, -1, 128))
        linear_184 = None
        value_states_26 = view_81.transpose(1, 2)
        view_81 = None
        cos_29 = cos_2.unsqueeze(1)
        sin_29 = sin_2.unsqueeze(1)
        mul_239 = query_states_26 * cos_29
        x1_52 = query_states_26[(Ellipsis, slice(None, 64, None))]
        x2_52 = query_states_26[(Ellipsis, slice(64, None, None))]
        query_states_26 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_53 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_240 = cat_53 * sin_29
        cat_53 = None
        q_embed_26 = mul_239 + mul_240
        mul_239 = mul_240 = None
        mul_241 = key_states_26 * cos_29
        cos_29 = None
        x1_53 = key_states_26[(Ellipsis, slice(None, 64, None))]
        x2_53 = key_states_26[(Ellipsis, slice(64, None, None))]
        key_states_26 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_54 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_242 = cat_54 * sin_29
        cat_54 = sin_29 = None
        k_embed_26 = mul_241 + mul_242
        mul_241 = mul_242 = None
        getitem_191 = k_embed_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_263 = getitem_191.expand(1, 2, 8, 2, 128)
        getitem_191 = None
        key_52 = hidden_states_263.reshape(1, 16, 2, 128)
        hidden_states_263 = None
        getitem_192 = value_states_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_264 = getitem_192.expand(1, 2, 8, 2, 128)
        getitem_192 = None
        value_52 = hidden_states_264.reshape(1, 16, 2, 128)
        hidden_states_264 = None
        attention_mask_27 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_26 = q_embed_26.contiguous()
        q_embed_26 = None
        key_53 = key_52.contiguous()
        key_52 = None
        value_53 = value_52.contiguous()
        value_52 = None
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_53,
            value_53,
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_26 = key_53 = value_53 = attention_mask_27 = None
        transpose_108 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_108.contiguous()
        transpose_108 = None
        reshape_80 = attn_output_105.reshape(1, 2, -1)
        attn_output_105 = None
        attn_output_106 = reshape_80.contiguous()
        reshape_80 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_265 = hidden_states_259 + attn_output_107
        hidden_states_259 = attn_output_107 = None
        hidden_states_266 = hidden_states_265.to(torch.float32)
        pow_54 = hidden_states_266.pow(2)
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_160 = variance_53 + 1e-06
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_267 = hidden_states_266 * rsqrt_53
        hidden_states_266 = rsqrt_53 = None
        to_111 = hidden_states_267.to(torch.bfloat16)
        hidden_states_267 = None
        hidden_states_268 = (
            l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
            * to_111
        )
        l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = (
            to_111
        ) = None
        linear_186 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_186, inplace=False)
        linear_186 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_268 = l_self_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_245 = silu_26 * linear_187
        silu_26 = linear_187 = None
        down_proj_26 = torch._C._nn.linear(
            mul_245,
            l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_245 = l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_269 = hidden_states_265 + down_proj_26
        hidden_states_265 = down_proj_26 = None
        hidden_states_270 = hidden_states_269.to(torch.float32)
        pow_55 = hidden_states_270.pow(2)
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_162 = variance_54 + 1e-06
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_162)
        add_162 = None
        hidden_states_271 = hidden_states_270 * rsqrt_54
        hidden_states_270 = rsqrt_54 = None
        to_113 = hidden_states_271.to(torch.bfloat16)
        hidden_states_271 = None
        hidden_states_272 = (
            l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
            * to_113
        )
        l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            to_113
        ) = None
        linear_189 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_82 = linear_189.view((1, 2, -1, 128))
        linear_189 = None
        query_states_27 = view_82.transpose(1, 2)
        view_82 = None
        linear_190 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_83 = linear_190.view((1, 2, -1, 128))
        linear_190 = None
        key_states_27 = view_83.transpose(1, 2)
        view_83 = None
        linear_191 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_272 = l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_84 = linear_191.view((1, 2, -1, 128))
        linear_191 = None
        value_states_27 = view_84.transpose(1, 2)
        view_84 = None
        cos_30 = cos_2.unsqueeze(1)
        sin_30 = sin_2.unsqueeze(1)
        mul_248 = query_states_27 * cos_30
        x1_54 = query_states_27[(Ellipsis, slice(None, 64, None))]
        x2_54 = query_states_27[(Ellipsis, slice(64, None, None))]
        query_states_27 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_55 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_249 = cat_55 * sin_30
        cat_55 = None
        q_embed_27 = mul_248 + mul_249
        mul_248 = mul_249 = None
        mul_250 = key_states_27 * cos_30
        cos_30 = None
        x1_55 = key_states_27[(Ellipsis, slice(None, 64, None))]
        x2_55 = key_states_27[(Ellipsis, slice(64, None, None))]
        key_states_27 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_56 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_251 = cat_56 * sin_30
        cat_56 = sin_30 = None
        k_embed_27 = mul_250 + mul_251
        mul_250 = mul_251 = None
        getitem_198 = k_embed_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_273 = getitem_198.expand(1, 2, 8, 2, 128)
        getitem_198 = None
        key_54 = hidden_states_273.reshape(1, 16, 2, 128)
        hidden_states_273 = None
        getitem_199 = value_states_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_274 = getitem_199.expand(1, 2, 8, 2, 128)
        getitem_199 = None
        value_54 = hidden_states_274.reshape(1, 16, 2, 128)
        hidden_states_274 = None
        attention_mask_28 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_27 = q_embed_27.contiguous()
        q_embed_27 = None
        key_55 = key_54.contiguous()
        key_54 = None
        value_55 = value_54.contiguous()
        value_54 = None
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_55,
            value_55,
            attn_mask=attention_mask_28,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_27 = key_55 = value_55 = attention_mask_28 = None
        transpose_112 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_112.contiguous()
        transpose_112 = None
        reshape_83 = attn_output_109.reshape(1, 2, -1)
        attn_output_109 = None
        attn_output_110 = reshape_83.contiguous()
        reshape_83 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_275 = hidden_states_269 + attn_output_111
        hidden_states_269 = attn_output_111 = None
        hidden_states_276 = hidden_states_275.to(torch.float32)
        pow_56 = hidden_states_276.pow(2)
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_166 = variance_55 + 1e-06
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_277 = hidden_states_276 * rsqrt_55
        hidden_states_276 = rsqrt_55 = None
        to_115 = hidden_states_277.to(torch.bfloat16)
        hidden_states_277 = None
        hidden_states_278 = (
            l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
            * to_115
        )
        l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = (
            to_115
        ) = None
        linear_193 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_193, inplace=False)
        linear_193 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_278 = l_self_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_254 = silu_27 * linear_194
        silu_27 = linear_194 = None
        down_proj_27 = torch._C._nn.linear(
            mul_254,
            l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_254 = l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_279 = hidden_states_275 + down_proj_27
        hidden_states_275 = down_proj_27 = None
        hidden_states_280 = hidden_states_279.to(torch.float32)
        pow_57 = hidden_states_280.pow(2)
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_168 = variance_56 + 1e-06
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_281 = hidden_states_280 * rsqrt_56
        hidden_states_280 = rsqrt_56 = None
        to_117 = hidden_states_281.to(torch.bfloat16)
        hidden_states_281 = None
        hidden_states_282 = (
            l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
            * to_117
        )
        l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = (
            to_117
        ) = None
        linear_196 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_85 = linear_196.view((1, 2, -1, 128))
        linear_196 = None
        query_states_28 = view_85.transpose(1, 2)
        view_85 = None
        linear_197 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_86 = linear_197.view((1, 2, -1, 128))
        linear_197 = None
        key_states_28 = view_86.transpose(1, 2)
        view_86 = None
        linear_198 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_282 = l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_87 = linear_198.view((1, 2, -1, 128))
        linear_198 = None
        value_states_28 = view_87.transpose(1, 2)
        view_87 = None
        cos_31 = cos_2.unsqueeze(1)
        sin_31 = sin_2.unsqueeze(1)
        mul_257 = query_states_28 * cos_31
        x1_56 = query_states_28[(Ellipsis, slice(None, 64, None))]
        x2_56 = query_states_28[(Ellipsis, slice(64, None, None))]
        query_states_28 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_57 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_258 = cat_57 * sin_31
        cat_57 = None
        q_embed_28 = mul_257 + mul_258
        mul_257 = mul_258 = None
        mul_259 = key_states_28 * cos_31
        cos_31 = None
        x1_57 = key_states_28[(Ellipsis, slice(None, 64, None))]
        x2_57 = key_states_28[(Ellipsis, slice(64, None, None))]
        key_states_28 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_58 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_260 = cat_58 * sin_31
        cat_58 = sin_31 = None
        k_embed_28 = mul_259 + mul_260
        mul_259 = mul_260 = None
        getitem_205 = k_embed_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_283 = getitem_205.expand(1, 2, 8, 2, 128)
        getitem_205 = None
        key_56 = hidden_states_283.reshape(1, 16, 2, 128)
        hidden_states_283 = None
        getitem_206 = value_states_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_284 = getitem_206.expand(1, 2, 8, 2, 128)
        getitem_206 = None
        value_56 = hidden_states_284.reshape(1, 16, 2, 128)
        hidden_states_284 = None
        attention_mask_29 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_28 = q_embed_28.contiguous()
        q_embed_28 = None
        key_57 = key_56.contiguous()
        key_56 = None
        value_57 = value_56.contiguous()
        value_56 = None
        attn_output_112 = torch._C._nn.scaled_dot_product_attention(
            query_28,
            key_57,
            value_57,
            attn_mask=attention_mask_29,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_28 = key_57 = value_57 = attention_mask_29 = None
        transpose_116 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_116.contiguous()
        transpose_116 = None
        reshape_86 = attn_output_113.reshape(1, 2, -1)
        attn_output_113 = None
        attn_output_114 = reshape_86.contiguous()
        reshape_86 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_285 = hidden_states_279 + attn_output_115
        hidden_states_279 = attn_output_115 = None
        hidden_states_286 = hidden_states_285.to(torch.float32)
        pow_58 = hidden_states_286.pow(2)
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_172 = variance_57 + 1e-06
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_172)
        add_172 = None
        hidden_states_287 = hidden_states_286 * rsqrt_57
        hidden_states_286 = rsqrt_57 = None
        to_119 = hidden_states_287.to(torch.bfloat16)
        hidden_states_287 = None
        hidden_states_288 = (
            l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
            * to_119
        )
        l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = (
            to_119
        ) = None
        linear_200 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_28 = torch.nn.functional.silu(linear_200, inplace=False)
        linear_200 = None
        linear_201 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_288 = l_self_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_263 = silu_28 * linear_201
        silu_28 = linear_201 = None
        down_proj_28 = torch._C._nn.linear(
            mul_263,
            l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_263 = l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_289 = hidden_states_285 + down_proj_28
        hidden_states_285 = down_proj_28 = None
        hidden_states_290 = hidden_states_289.to(torch.float32)
        pow_59 = hidden_states_290.pow(2)
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_174 = variance_58 + 1e-06
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_291 = hidden_states_290 * rsqrt_58
        hidden_states_290 = rsqrt_58 = None
        to_121 = hidden_states_291.to(torch.bfloat16)
        hidden_states_291 = None
        hidden_states_292 = (
            l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
            * to_121
        )
        l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = (
            to_121
        ) = None
        linear_203 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_88 = linear_203.view((1, 2, -1, 128))
        linear_203 = None
        query_states_29 = view_88.transpose(1, 2)
        view_88 = None
        linear_204 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_89 = linear_204.view((1, 2, -1, 128))
        linear_204 = None
        key_states_29 = view_89.transpose(1, 2)
        view_89 = None
        linear_205 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_292 = l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_90 = linear_205.view((1, 2, -1, 128))
        linear_205 = None
        value_states_29 = view_90.transpose(1, 2)
        view_90 = None
        cos_32 = cos_2.unsqueeze(1)
        sin_32 = sin_2.unsqueeze(1)
        mul_266 = query_states_29 * cos_32
        x1_58 = query_states_29[(Ellipsis, slice(None, 64, None))]
        x2_58 = query_states_29[(Ellipsis, slice(64, None, None))]
        query_states_29 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_59 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_267 = cat_59 * sin_32
        cat_59 = None
        q_embed_29 = mul_266 + mul_267
        mul_266 = mul_267 = None
        mul_268 = key_states_29 * cos_32
        cos_32 = None
        x1_59 = key_states_29[(Ellipsis, slice(None, 64, None))]
        x2_59 = key_states_29[(Ellipsis, slice(64, None, None))]
        key_states_29 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_60 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_269 = cat_60 * sin_32
        cat_60 = sin_32 = None
        k_embed_29 = mul_268 + mul_269
        mul_268 = mul_269 = None
        getitem_212 = k_embed_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_293 = getitem_212.expand(1, 2, 8, 2, 128)
        getitem_212 = None
        key_58 = hidden_states_293.reshape(1, 16, 2, 128)
        hidden_states_293 = None
        getitem_213 = value_states_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_294 = getitem_213.expand(1, 2, 8, 2, 128)
        getitem_213 = None
        value_58 = hidden_states_294.reshape(1, 16, 2, 128)
        hidden_states_294 = None
        attention_mask_30 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_29 = q_embed_29.contiguous()
        q_embed_29 = None
        key_59 = key_58.contiguous()
        key_58 = None
        value_59 = value_58.contiguous()
        value_58 = None
        attn_output_116 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_59,
            value_59,
            attn_mask=attention_mask_30,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_29 = key_59 = value_59 = attention_mask_30 = None
        transpose_120 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_120.contiguous()
        transpose_120 = None
        reshape_89 = attn_output_117.reshape(1, 2, -1)
        attn_output_117 = None
        attn_output_118 = reshape_89.contiguous()
        reshape_89 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_295 = hidden_states_289 + attn_output_119
        hidden_states_289 = attn_output_119 = None
        hidden_states_296 = hidden_states_295.to(torch.float32)
        pow_60 = hidden_states_296.pow(2)
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_178 = variance_59 + 1e-06
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_178)
        add_178 = None
        hidden_states_297 = hidden_states_296 * rsqrt_59
        hidden_states_296 = rsqrt_59 = None
        to_123 = hidden_states_297.to(torch.bfloat16)
        hidden_states_297 = None
        hidden_states_298 = (
            l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
            * to_123
        )
        l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = (
            to_123
        ) = None
        linear_207 = torch._C._nn.linear(
            hidden_states_298,
            l_self_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_29 = torch.nn.functional.silu(linear_207, inplace=False)
        linear_207 = None
        linear_208 = torch._C._nn.linear(
            hidden_states_298,
            l_self_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_298 = l_self_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_272 = silu_29 * linear_208
        silu_29 = linear_208 = None
        down_proj_29 = torch._C._nn.linear(
            mul_272,
            l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_272 = l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_299 = hidden_states_295 + down_proj_29
        hidden_states_295 = down_proj_29 = None
        hidden_states_300 = hidden_states_299.to(torch.float32)
        pow_61 = hidden_states_300.pow(2)
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_180 = variance_60 + 1e-06
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_180)
        add_180 = None
        hidden_states_301 = hidden_states_300 * rsqrt_60
        hidden_states_300 = rsqrt_60 = None
        to_125 = hidden_states_301.to(torch.bfloat16)
        hidden_states_301 = None
        hidden_states_302 = (
            l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
            * to_125
        )
        l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = (
            to_125
        ) = None
        linear_210 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_91 = linear_210.view((1, 2, -1, 128))
        linear_210 = None
        query_states_30 = view_91.transpose(1, 2)
        view_91 = None
        linear_211 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_92 = linear_211.view((1, 2, -1, 128))
        linear_211 = None
        key_states_30 = view_92.transpose(1, 2)
        view_92 = None
        linear_212 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_302 = l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_93 = linear_212.view((1, 2, -1, 128))
        linear_212 = None
        value_states_30 = view_93.transpose(1, 2)
        view_93 = None
        cos_33 = cos_2.unsqueeze(1)
        sin_33 = sin_2.unsqueeze(1)
        mul_275 = query_states_30 * cos_33
        x1_60 = query_states_30[(Ellipsis, slice(None, 64, None))]
        x2_60 = query_states_30[(Ellipsis, slice(64, None, None))]
        query_states_30 = None
        neg_60 = -x2_60
        x2_60 = None
        cat_61 = torch.cat((neg_60, x1_60), dim=-1)
        neg_60 = x1_60 = None
        mul_276 = cat_61 * sin_33
        cat_61 = None
        q_embed_30 = mul_275 + mul_276
        mul_275 = mul_276 = None
        mul_277 = key_states_30 * cos_33
        cos_33 = None
        x1_61 = key_states_30[(Ellipsis, slice(None, 64, None))]
        x2_61 = key_states_30[(Ellipsis, slice(64, None, None))]
        key_states_30 = None
        neg_61 = -x2_61
        x2_61 = None
        cat_62 = torch.cat((neg_61, x1_61), dim=-1)
        neg_61 = x1_61 = None
        mul_278 = cat_62 * sin_33
        cat_62 = sin_33 = None
        k_embed_30 = mul_277 + mul_278
        mul_277 = mul_278 = None
        getitem_219 = k_embed_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_303 = getitem_219.expand(1, 2, 8, 2, 128)
        getitem_219 = None
        key_60 = hidden_states_303.reshape(1, 16, 2, 128)
        hidden_states_303 = None
        getitem_220 = value_states_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_304 = getitem_220.expand(1, 2, 8, 2, 128)
        getitem_220 = None
        value_60 = hidden_states_304.reshape(1, 16, 2, 128)
        hidden_states_304 = None
        attention_mask_31 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_30 = q_embed_30.contiguous()
        q_embed_30 = None
        key_61 = key_60.contiguous()
        key_60 = None
        value_61 = value_60.contiguous()
        value_60 = None
        attn_output_120 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_61,
            value_61,
            attn_mask=attention_mask_31,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_30 = key_61 = value_61 = attention_mask_31 = None
        transpose_124 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_124.contiguous()
        transpose_124 = None
        reshape_92 = attn_output_121.reshape(1, 2, -1)
        attn_output_121 = None
        attn_output_122 = reshape_92.contiguous()
        reshape_92 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_305 = hidden_states_299 + attn_output_123
        hidden_states_299 = attn_output_123 = None
        hidden_states_306 = hidden_states_305.to(torch.float32)
        pow_62 = hidden_states_306.pow(2)
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_184 = variance_61 + 1e-06
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_307 = hidden_states_306 * rsqrt_61
        hidden_states_306 = rsqrt_61 = None
        to_127 = hidden_states_307.to(torch.bfloat16)
        hidden_states_307 = None
        hidden_states_308 = (
            l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
            * to_127
        )
        l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = (
            to_127
        ) = None
        linear_214 = torch._C._nn.linear(
            hidden_states_308,
            l_self_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_30 = torch.nn.functional.silu(linear_214, inplace=False)
        linear_214 = None
        linear_215 = torch._C._nn.linear(
            hidden_states_308,
            l_self_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_308 = l_self_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_281 = silu_30 * linear_215
        silu_30 = linear_215 = None
        down_proj_30 = torch._C._nn.linear(
            mul_281,
            l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_281 = l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_309 = hidden_states_305 + down_proj_30
        hidden_states_305 = down_proj_30 = None
        hidden_states_310 = hidden_states_309.to(torch.float32)
        pow_63 = hidden_states_310.pow(2)
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_186 = variance_62 + 1e-06
        variance_62 = None
        rsqrt_62 = torch.rsqrt(add_186)
        add_186 = None
        hidden_states_311 = hidden_states_310 * rsqrt_62
        hidden_states_310 = rsqrt_62 = None
        to_129 = hidden_states_311.to(torch.bfloat16)
        hidden_states_311 = None
        hidden_states_312 = (
            l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
            * to_129
        )
        l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = (
            to_129
        ) = None
        linear_217 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_94 = linear_217.view((1, 2, -1, 128))
        linear_217 = None
        query_states_31 = view_94.transpose(1, 2)
        view_94 = None
        linear_218 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_95 = linear_218.view((1, 2, -1, 128))
        linear_218 = None
        key_states_31 = view_95.transpose(1, 2)
        view_95 = None
        linear_219 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_312 = l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_96 = linear_219.view((1, 2, -1, 128))
        linear_219 = None
        value_states_31 = view_96.transpose(1, 2)
        view_96 = None
        cos_34 = cos_2.unsqueeze(1)
        sin_34 = sin_2.unsqueeze(1)
        mul_284 = query_states_31 * cos_34
        x1_62 = query_states_31[(Ellipsis, slice(None, 64, None))]
        x2_62 = query_states_31[(Ellipsis, slice(64, None, None))]
        query_states_31 = None
        neg_62 = -x2_62
        x2_62 = None
        cat_63 = torch.cat((neg_62, x1_62), dim=-1)
        neg_62 = x1_62 = None
        mul_285 = cat_63 * sin_34
        cat_63 = None
        q_embed_31 = mul_284 + mul_285
        mul_284 = mul_285 = None
        mul_286 = key_states_31 * cos_34
        cos_34 = None
        x1_63 = key_states_31[(Ellipsis, slice(None, 64, None))]
        x2_63 = key_states_31[(Ellipsis, slice(64, None, None))]
        key_states_31 = None
        neg_63 = -x2_63
        x2_63 = None
        cat_64 = torch.cat((neg_63, x1_63), dim=-1)
        neg_63 = x1_63 = None
        mul_287 = cat_64 * sin_34
        cat_64 = sin_34 = None
        k_embed_31 = mul_286 + mul_287
        mul_286 = mul_287 = None
        getitem_226 = k_embed_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_313 = getitem_226.expand(1, 2, 8, 2, 128)
        getitem_226 = None
        key_62 = hidden_states_313.reshape(1, 16, 2, 128)
        hidden_states_313 = None
        getitem_227 = value_states_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_314 = getitem_227.expand(1, 2, 8, 2, 128)
        getitem_227 = None
        value_62 = hidden_states_314.reshape(1, 16, 2, 128)
        hidden_states_314 = None
        attention_mask_32 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_31 = q_embed_31.contiguous()
        q_embed_31 = None
        key_63 = key_62.contiguous()
        key_62 = None
        value_63 = value_62.contiguous()
        value_62 = None
        attn_output_124 = torch._C._nn.scaled_dot_product_attention(
            query_31,
            key_63,
            value_63,
            attn_mask=attention_mask_32,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_31 = key_63 = value_63 = attention_mask_32 = None
        transpose_128 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_128.contiguous()
        transpose_128 = None
        reshape_95 = attn_output_125.reshape(1, 2, -1)
        attn_output_125 = None
        attn_output_126 = reshape_95.contiguous()
        reshape_95 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_315 = hidden_states_309 + attn_output_127
        hidden_states_309 = attn_output_127 = None
        hidden_states_316 = hidden_states_315.to(torch.float32)
        pow_64 = hidden_states_316.pow(2)
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_190 = variance_63 + 1e-06
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_317 = hidden_states_316 * rsqrt_63
        hidden_states_316 = rsqrt_63 = None
        to_131 = hidden_states_317.to(torch.bfloat16)
        hidden_states_317 = None
        hidden_states_318 = (
            l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
            * to_131
        )
        l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = (
            to_131
        ) = None
        linear_221 = torch._C._nn.linear(
            hidden_states_318,
            l_self_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_31 = torch.nn.functional.silu(linear_221, inplace=False)
        linear_221 = None
        linear_222 = torch._C._nn.linear(
            hidden_states_318,
            l_self_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_318 = l_self_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_290 = silu_31 * linear_222
        silu_31 = linear_222 = None
        down_proj_31 = torch._C._nn.linear(
            mul_290,
            l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_290 = l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_319 = hidden_states_315 + down_proj_31
        hidden_states_315 = down_proj_31 = None
        hidden_states_320 = hidden_states_319.to(torch.float32)
        pow_65 = hidden_states_320.pow(2)
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_192 = variance_64 + 1e-06
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_321 = hidden_states_320 * rsqrt_64
        hidden_states_320 = rsqrt_64 = None
        to_133 = hidden_states_321.to(torch.bfloat16)
        hidden_states_321 = None
        hidden_states_322 = (
            l_self_modules_layers_modules_32_modules_input_layernorm_parameters_weight_
            * to_133
        )
        l_self_modules_layers_modules_32_modules_input_layernorm_parameters_weight_ = (
            to_133
        ) = None
        linear_224 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_97 = linear_224.view((1, 2, -1, 128))
        linear_224 = None
        query_states_32 = view_97.transpose(1, 2)
        view_97 = None
        linear_225 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_98 = linear_225.view((1, 2, -1, 128))
        linear_225 = None
        key_states_32 = view_98.transpose(1, 2)
        view_98 = None
        linear_226 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_322 = l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_99 = linear_226.view((1, 2, -1, 128))
        linear_226 = None
        value_states_32 = view_99.transpose(1, 2)
        view_99 = None
        cos_35 = cos_2.unsqueeze(1)
        sin_35 = sin_2.unsqueeze(1)
        mul_293 = query_states_32 * cos_35
        x1_64 = query_states_32[(Ellipsis, slice(None, 64, None))]
        x2_64 = query_states_32[(Ellipsis, slice(64, None, None))]
        query_states_32 = None
        neg_64 = -x2_64
        x2_64 = None
        cat_65 = torch.cat((neg_64, x1_64), dim=-1)
        neg_64 = x1_64 = None
        mul_294 = cat_65 * sin_35
        cat_65 = None
        q_embed_32 = mul_293 + mul_294
        mul_293 = mul_294 = None
        mul_295 = key_states_32 * cos_35
        cos_35 = None
        x1_65 = key_states_32[(Ellipsis, slice(None, 64, None))]
        x2_65 = key_states_32[(Ellipsis, slice(64, None, None))]
        key_states_32 = None
        neg_65 = -x2_65
        x2_65 = None
        cat_66 = torch.cat((neg_65, x1_65), dim=-1)
        neg_65 = x1_65 = None
        mul_296 = cat_66 * sin_35
        cat_66 = sin_35 = None
        k_embed_32 = mul_295 + mul_296
        mul_295 = mul_296 = None
        getitem_233 = k_embed_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_323 = getitem_233.expand(1, 2, 8, 2, 128)
        getitem_233 = None
        key_64 = hidden_states_323.reshape(1, 16, 2, 128)
        hidden_states_323 = None
        getitem_234 = value_states_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_324 = getitem_234.expand(1, 2, 8, 2, 128)
        getitem_234 = None
        value_64 = hidden_states_324.reshape(1, 16, 2, 128)
        hidden_states_324 = None
        attention_mask_33 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_32 = q_embed_32.contiguous()
        q_embed_32 = None
        key_65 = key_64.contiguous()
        key_64 = None
        value_65 = value_64.contiguous()
        value_64 = None
        attn_output_128 = torch._C._nn.scaled_dot_product_attention(
            query_32,
            key_65,
            value_65,
            attn_mask=attention_mask_33,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_32 = key_65 = value_65 = attention_mask_33 = None
        transpose_132 = attn_output_128.transpose(1, 2)
        attn_output_128 = None
        attn_output_129 = transpose_132.contiguous()
        transpose_132 = None
        reshape_98 = attn_output_129.reshape(1, 2, -1)
        attn_output_129 = None
        attn_output_130 = reshape_98.contiguous()
        reshape_98 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_130 = l_self_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_325 = hidden_states_319 + attn_output_131
        hidden_states_319 = attn_output_131 = None
        hidden_states_326 = hidden_states_325.to(torch.float32)
        pow_66 = hidden_states_326.pow(2)
        variance_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_196 = variance_65 + 1e-06
        variance_65 = None
        rsqrt_65 = torch.rsqrt(add_196)
        add_196 = None
        hidden_states_327 = hidden_states_326 * rsqrt_65
        hidden_states_326 = rsqrt_65 = None
        to_135 = hidden_states_327.to(torch.bfloat16)
        hidden_states_327 = None
        hidden_states_328 = (
            l_self_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_
            * to_135
        )
        l_self_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_ = (
            to_135
        ) = None
        linear_228 = torch._C._nn.linear(
            hidden_states_328,
            l_self_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_32 = torch.nn.functional.silu(linear_228, inplace=False)
        linear_228 = None
        linear_229 = torch._C._nn.linear(
            hidden_states_328,
            l_self_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_328 = l_self_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_299 = silu_32 * linear_229
        silu_32 = linear_229 = None
        down_proj_32 = torch._C._nn.linear(
            mul_299,
            l_self_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_299 = l_self_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_329 = hidden_states_325 + down_proj_32
        hidden_states_325 = down_proj_32 = None
        hidden_states_330 = hidden_states_329.to(torch.float32)
        pow_67 = hidden_states_330.pow(2)
        variance_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_198 = variance_66 + 1e-06
        variance_66 = None
        rsqrt_66 = torch.rsqrt(add_198)
        add_198 = None
        hidden_states_331 = hidden_states_330 * rsqrt_66
        hidden_states_330 = rsqrt_66 = None
        to_137 = hidden_states_331.to(torch.bfloat16)
        hidden_states_331 = None
        hidden_states_332 = (
            l_self_modules_layers_modules_33_modules_input_layernorm_parameters_weight_
            * to_137
        )
        l_self_modules_layers_modules_33_modules_input_layernorm_parameters_weight_ = (
            to_137
        ) = None
        linear_231 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_100 = linear_231.view((1, 2, -1, 128))
        linear_231 = None
        query_states_33 = view_100.transpose(1, 2)
        view_100 = None
        linear_232 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_101 = linear_232.view((1, 2, -1, 128))
        linear_232 = None
        key_states_33 = view_101.transpose(1, 2)
        view_101 = None
        linear_233 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_332 = l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_102 = linear_233.view((1, 2, -1, 128))
        linear_233 = None
        value_states_33 = view_102.transpose(1, 2)
        view_102 = None
        cos_36 = cos_2.unsqueeze(1)
        sin_36 = sin_2.unsqueeze(1)
        mul_302 = query_states_33 * cos_36
        x1_66 = query_states_33[(Ellipsis, slice(None, 64, None))]
        x2_66 = query_states_33[(Ellipsis, slice(64, None, None))]
        query_states_33 = None
        neg_66 = -x2_66
        x2_66 = None
        cat_67 = torch.cat((neg_66, x1_66), dim=-1)
        neg_66 = x1_66 = None
        mul_303 = cat_67 * sin_36
        cat_67 = None
        q_embed_33 = mul_302 + mul_303
        mul_302 = mul_303 = None
        mul_304 = key_states_33 * cos_36
        cos_36 = None
        x1_67 = key_states_33[(Ellipsis, slice(None, 64, None))]
        x2_67 = key_states_33[(Ellipsis, slice(64, None, None))]
        key_states_33 = None
        neg_67 = -x2_67
        x2_67 = None
        cat_68 = torch.cat((neg_67, x1_67), dim=-1)
        neg_67 = x1_67 = None
        mul_305 = cat_68 * sin_36
        cat_68 = sin_36 = None
        k_embed_33 = mul_304 + mul_305
        mul_304 = mul_305 = None
        getitem_240 = k_embed_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_333 = getitem_240.expand(1, 2, 8, 2, 128)
        getitem_240 = None
        key_66 = hidden_states_333.reshape(1, 16, 2, 128)
        hidden_states_333 = None
        getitem_241 = value_states_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_334 = getitem_241.expand(1, 2, 8, 2, 128)
        getitem_241 = None
        value_66 = hidden_states_334.reshape(1, 16, 2, 128)
        hidden_states_334 = None
        attention_mask_34 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_33 = q_embed_33.contiguous()
        q_embed_33 = None
        key_67 = key_66.contiguous()
        key_66 = None
        value_67 = value_66.contiguous()
        value_66 = None
        attn_output_132 = torch._C._nn.scaled_dot_product_attention(
            query_33,
            key_67,
            value_67,
            attn_mask=attention_mask_34,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_33 = key_67 = value_67 = attention_mask_34 = None
        transpose_136 = attn_output_132.transpose(1, 2)
        attn_output_132 = None
        attn_output_133 = transpose_136.contiguous()
        transpose_136 = None
        reshape_101 = attn_output_133.reshape(1, 2, -1)
        attn_output_133 = None
        attn_output_134 = reshape_101.contiguous()
        reshape_101 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_134 = l_self_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_335 = hidden_states_329 + attn_output_135
        hidden_states_329 = attn_output_135 = None
        hidden_states_336 = hidden_states_335.to(torch.float32)
        pow_68 = hidden_states_336.pow(2)
        variance_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_202 = variance_67 + 1e-06
        variance_67 = None
        rsqrt_67 = torch.rsqrt(add_202)
        add_202 = None
        hidden_states_337 = hidden_states_336 * rsqrt_67
        hidden_states_336 = rsqrt_67 = None
        to_139 = hidden_states_337.to(torch.bfloat16)
        hidden_states_337 = None
        hidden_states_338 = (
            l_self_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_
            * to_139
        )
        l_self_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_ = (
            to_139
        ) = None
        linear_235 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_33 = torch.nn.functional.silu(linear_235, inplace=False)
        linear_235 = None
        linear_236 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_338 = l_self_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_308 = silu_33 * linear_236
        silu_33 = linear_236 = None
        down_proj_33 = torch._C._nn.linear(
            mul_308,
            l_self_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_308 = l_self_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_339 = hidden_states_335 + down_proj_33
        hidden_states_335 = down_proj_33 = None
        hidden_states_340 = hidden_states_339.to(torch.float32)
        pow_69 = hidden_states_340.pow(2)
        variance_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_204 = variance_68 + 1e-06
        variance_68 = None
        rsqrt_68 = torch.rsqrt(add_204)
        add_204 = None
        hidden_states_341 = hidden_states_340 * rsqrt_68
        hidden_states_340 = rsqrt_68 = None
        to_141 = hidden_states_341.to(torch.bfloat16)
        hidden_states_341 = None
        hidden_states_342 = (
            l_self_modules_layers_modules_34_modules_input_layernorm_parameters_weight_
            * to_141
        )
        l_self_modules_layers_modules_34_modules_input_layernorm_parameters_weight_ = (
            to_141
        ) = None
        linear_238 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_103 = linear_238.view((1, 2, -1, 128))
        linear_238 = None
        query_states_34 = view_103.transpose(1, 2)
        view_103 = None
        linear_239 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_104 = linear_239.view((1, 2, -1, 128))
        linear_239 = None
        key_states_34 = view_104.transpose(1, 2)
        view_104 = None
        linear_240 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_342 = l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_105 = linear_240.view((1, 2, -1, 128))
        linear_240 = None
        value_states_34 = view_105.transpose(1, 2)
        view_105 = None
        cos_37 = cos_2.unsqueeze(1)
        sin_37 = sin_2.unsqueeze(1)
        mul_311 = query_states_34 * cos_37
        x1_68 = query_states_34[(Ellipsis, slice(None, 64, None))]
        x2_68 = query_states_34[(Ellipsis, slice(64, None, None))]
        query_states_34 = None
        neg_68 = -x2_68
        x2_68 = None
        cat_69 = torch.cat((neg_68, x1_68), dim=-1)
        neg_68 = x1_68 = None
        mul_312 = cat_69 * sin_37
        cat_69 = None
        q_embed_34 = mul_311 + mul_312
        mul_311 = mul_312 = None
        mul_313 = key_states_34 * cos_37
        cos_37 = None
        x1_69 = key_states_34[(Ellipsis, slice(None, 64, None))]
        x2_69 = key_states_34[(Ellipsis, slice(64, None, None))]
        key_states_34 = None
        neg_69 = -x2_69
        x2_69 = None
        cat_70 = torch.cat((neg_69, x1_69), dim=-1)
        neg_69 = x1_69 = None
        mul_314 = cat_70 * sin_37
        cat_70 = sin_37 = None
        k_embed_34 = mul_313 + mul_314
        mul_313 = mul_314 = None
        getitem_247 = k_embed_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_343 = getitem_247.expand(1, 2, 8, 2, 128)
        getitem_247 = None
        key_68 = hidden_states_343.reshape(1, 16, 2, 128)
        hidden_states_343 = None
        getitem_248 = value_states_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_344 = getitem_248.expand(1, 2, 8, 2, 128)
        getitem_248 = None
        value_68 = hidden_states_344.reshape(1, 16, 2, 128)
        hidden_states_344 = None
        attention_mask_35 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_34 = q_embed_34.contiguous()
        q_embed_34 = None
        key_69 = key_68.contiguous()
        key_68 = None
        value_69 = value_68.contiguous()
        value_68 = None
        attn_output_136 = torch._C._nn.scaled_dot_product_attention(
            query_34,
            key_69,
            value_69,
            attn_mask=attention_mask_35,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_34 = key_69 = value_69 = attention_mask_35 = None
        transpose_140 = attn_output_136.transpose(1, 2)
        attn_output_136 = None
        attn_output_137 = transpose_140.contiguous()
        transpose_140 = None
        reshape_104 = attn_output_137.reshape(1, 2, -1)
        attn_output_137 = None
        attn_output_138 = reshape_104.contiguous()
        reshape_104 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_138 = l_self_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_345 = hidden_states_339 + attn_output_139
        hidden_states_339 = attn_output_139 = None
        hidden_states_346 = hidden_states_345.to(torch.float32)
        pow_70 = hidden_states_346.pow(2)
        variance_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_208 = variance_69 + 1e-06
        variance_69 = None
        rsqrt_69 = torch.rsqrt(add_208)
        add_208 = None
        hidden_states_347 = hidden_states_346 * rsqrt_69
        hidden_states_346 = rsqrt_69 = None
        to_143 = hidden_states_347.to(torch.bfloat16)
        hidden_states_347 = None
        hidden_states_348 = (
            l_self_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_
            * to_143
        )
        l_self_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_ = (
            to_143
        ) = None
        linear_242 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_34 = torch.nn.functional.silu(linear_242, inplace=False)
        linear_242 = None
        linear_243 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_348 = l_self_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_317 = silu_34 * linear_243
        silu_34 = linear_243 = None
        down_proj_34 = torch._C._nn.linear(
            mul_317,
            l_self_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_317 = l_self_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_349 = hidden_states_345 + down_proj_34
        hidden_states_345 = down_proj_34 = None
        hidden_states_350 = hidden_states_349.to(torch.float32)
        pow_71 = hidden_states_350.pow(2)
        variance_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_210 = variance_70 + 1e-06
        variance_70 = None
        rsqrt_70 = torch.rsqrt(add_210)
        add_210 = None
        hidden_states_351 = hidden_states_350 * rsqrt_70
        hidden_states_350 = rsqrt_70 = None
        to_145 = hidden_states_351.to(torch.bfloat16)
        hidden_states_351 = None
        hidden_states_352 = (
            l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_
            * to_145
        )
        l_self_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = (
            to_145
        ) = None
        linear_245 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_106 = linear_245.view((1, 2, -1, 128))
        linear_245 = None
        query_states_35 = view_106.transpose(1, 2)
        view_106 = None
        linear_246 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_107 = linear_246.view((1, 2, -1, 128))
        linear_246 = None
        key_states_35 = view_107.transpose(1, 2)
        view_107 = None
        linear_247 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_352 = l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_108 = linear_247.view((1, 2, -1, 128))
        linear_247 = None
        value_states_35 = view_108.transpose(1, 2)
        view_108 = None
        cos_38 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_38 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_320 = query_states_35 * cos_38
        x1_70 = query_states_35[(Ellipsis, slice(None, 64, None))]
        x2_70 = query_states_35[(Ellipsis, slice(64, None, None))]
        query_states_35 = None
        neg_70 = -x2_70
        x2_70 = None
        cat_71 = torch.cat((neg_70, x1_70), dim=-1)
        neg_70 = x1_70 = None
        mul_321 = cat_71 * sin_38
        cat_71 = None
        q_embed_35 = mul_320 + mul_321
        mul_320 = mul_321 = None
        mul_322 = key_states_35 * cos_38
        cos_38 = None
        x1_71 = key_states_35[(Ellipsis, slice(None, 64, None))]
        x2_71 = key_states_35[(Ellipsis, slice(64, None, None))]
        key_states_35 = None
        neg_71 = -x2_71
        x2_71 = None
        cat_72 = torch.cat((neg_71, x1_71), dim=-1)
        neg_71 = x1_71 = None
        mul_323 = cat_72 * sin_38
        cat_72 = sin_38 = None
        k_embed_35 = mul_322 + mul_323
        mul_322 = mul_323 = None
        getitem_254 = k_embed_35[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_353 = getitem_254.expand(1, 2, 8, 2, 128)
        getitem_254 = None
        key_70 = hidden_states_353.reshape(1, 16, 2, 128)
        hidden_states_353 = None
        getitem_255 = value_states_35[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_354 = getitem_255.expand(1, 2, 8, 2, 128)
        getitem_255 = None
        value_70 = hidden_states_354.reshape(1, 16, 2, 128)
        hidden_states_354 = None
        attention_mask_36 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        causal_mask_2 = None
        query_35 = q_embed_35.contiguous()
        q_embed_35 = None
        key_71 = key_70.contiguous()
        key_70 = None
        value_71 = value_70.contiguous()
        value_70 = None
        attn_output_140 = torch._C._nn.scaled_dot_product_attention(
            query_35,
            key_71,
            value_71,
            attn_mask=attention_mask_36,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_35 = key_71 = value_71 = attention_mask_36 = None
        transpose_144 = attn_output_140.transpose(1, 2)
        attn_output_140 = None
        attn_output_141 = transpose_144.contiguous()
        transpose_144 = None
        reshape_107 = attn_output_141.reshape(1, 2, -1)
        attn_output_141 = None
        attn_output_142 = reshape_107.contiguous()
        reshape_107 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_142 = l_self_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_355 = hidden_states_349 + attn_output_143
        hidden_states_349 = attn_output_143 = None
        hidden_states_356 = hidden_states_355.to(torch.float32)
        pow_72 = hidden_states_356.pow(2)
        variance_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_214 = variance_71 + 1e-06
        variance_71 = None
        rsqrt_71 = torch.rsqrt(add_214)
        add_214 = None
        hidden_states_357 = hidden_states_356 * rsqrt_71
        hidden_states_356 = rsqrt_71 = None
        to_147 = hidden_states_357.to(torch.bfloat16)
        hidden_states_357 = None
        hidden_states_358 = (
            l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_
            * to_147
        )
        l_self_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = (
            to_147
        ) = None
        linear_249 = torch._C._nn.linear(
            hidden_states_358,
            l_self_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_35 = torch.nn.functional.silu(linear_249, inplace=False)
        linear_249 = None
        linear_250 = torch._C._nn.linear(
            hidden_states_358,
            l_self_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_358 = l_self_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_326 = silu_35 * linear_250
        silu_35 = linear_250 = None
        down_proj_35 = torch._C._nn.linear(
            mul_326,
            l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_326 = l_self_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_359 = hidden_states_355 + down_proj_35
        hidden_states_355 = down_proj_35 = None
        hidden_states_360 = hidden_states_359.to(torch.float32)
        hidden_states_359 = None
        pow_73 = hidden_states_360.pow(2)
        variance_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_216 = variance_72 + 1e-06
        variance_72 = None
        rsqrt_72 = torch.rsqrt(add_216)
        add_216 = None
        hidden_states_361 = hidden_states_360 * rsqrt_72
        hidden_states_360 = rsqrt_72 = None
        to_149 = hidden_states_361.to(torch.bfloat16)
        hidden_states_361 = None
        hidden_states_362 = l_self_modules_norm_parameters_weight_ * to_149
        l_self_modules_norm_parameters_weight_ = to_149 = None
        return (
            value_states,
            k_embed,
            value_states_1,
            k_embed_1,
            value_states_2,
            k_embed_2,
            value_states_3,
            k_embed_3,
            value_states_4,
            k_embed_4,
            value_states_5,
            k_embed_5,
            value_states_6,
            k_embed_6,
            value_states_7,
            k_embed_7,
            value_states_8,
            k_embed_8,
            value_states_9,
            k_embed_9,
            value_states_10,
            k_embed_10,
            value_states_11,
            k_embed_11,
            value_states_12,
            k_embed_12,
            value_states_13,
            k_embed_13,
            value_states_14,
            k_embed_14,
            value_states_15,
            k_embed_15,
            value_states_16,
            k_embed_16,
            value_states_17,
            k_embed_17,
            value_states_18,
            k_embed_18,
            value_states_19,
            k_embed_19,
            value_states_20,
            k_embed_20,
            value_states_21,
            k_embed_21,
            value_states_22,
            k_embed_22,
            value_states_23,
            k_embed_23,
            value_states_24,
            k_embed_24,
            value_states_25,
            k_embed_25,
            value_states_26,
            k_embed_26,
            value_states_27,
            k_embed_27,
            value_states_28,
            k_embed_28,
            value_states_29,
            k_embed_29,
            value_states_30,
            k_embed_30,
            value_states_31,
            k_embed_31,
            value_states_32,
            k_embed_32,
            value_states_33,
            k_embed_33,
            value_states_34,
            k_embed_34,
            value_states_35,
            k_embed_35,
            hidden_states_362,
        )
