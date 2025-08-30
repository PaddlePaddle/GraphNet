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
        L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
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
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        _set_grad_enabled_1 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        hidden_states = l_inputs_embeds_.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-05
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        hidden_states = rsqrt = None
        to_3 = hidden_states_1.to(torch.bfloat16)
        hidden_states_1 = None
        hidden_states_2 = (
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
            * to_3
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            to_3
        ) = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_1 = linear.view((1, 2, -1, 128))
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_2 = linear_1.view((1, 2, -1, 128))
        linear_1 = None
        key_states = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_3 = linear_2.view((1, 2, -1, 128))
        linear_2 = None
        value_states = view_3.transpose(1, 2)
        view_3 = None
        cos_2 = cos_1.unsqueeze(1)
        sin_2 = sin_1.unsqueeze(1)
        getitem_5 = cos_2[(Ellipsis, slice(None, 64, None))]
        cos_2 = None
        cos_3 = getitem_5.repeat_interleave(2, dim=-1)
        getitem_5 = None
        getitem_6 = sin_2[(Ellipsis, slice(None, 64, None))]
        sin_2 = None
        sin_3 = getitem_6.repeat_interleave(2, dim=-1)
        getitem_6 = None
        float_5 = query_states.float()
        mul_5 = float_5 * cos_3
        float_5 = None
        x1 = query_states[(Ellipsis, slice(0, None, 2))]
        x2 = query_states[(Ellipsis, slice(1, None, 2))]
        query_states = None
        neg = -x2
        x2 = None
        stack = torch.stack((neg, x1), dim=-1)
        neg = x1 = None
        flatten = stack.flatten(-2)
        stack = None
        float_6 = flatten.float()
        flatten = None
        mul_6 = float_6 * sin_3
        float_6 = None
        q_embed = mul_5 + mul_6
        mul_5 = mul_6 = None
        float_7 = key_states.float()
        mul_7 = float_7 * cos_3
        float_7 = cos_3 = None
        x1_1 = key_states[(Ellipsis, slice(0, None, 2))]
        x2_1 = key_states[(Ellipsis, slice(1, None, 2))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        stack_1 = torch.stack((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        flatten_1 = stack_1.flatten(-2)
        stack_1 = None
        float_8 = flatten_1.float()
        flatten_1 = None
        mul_8 = float_8 * sin_3
        float_8 = sin_3 = None
        k_embed = mul_7 + mul_8
        mul_7 = mul_8 = None
        query_states_1 = q_embed.to(torch.bfloat16)
        q_embed = None
        key_states_1 = k_embed.to(torch.bfloat16)
        k_embed = None
        getitem_11 = key_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_3 = getitem_11.expand(1, 2, 8, 2, 128)
        getitem_11 = None
        key = hidden_states_3.reshape(1, 16, 2, 128)
        hidden_states_3 = None
        getitem_12 = value_states[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_4 = getitem_12.expand(1, 2, 8, 2, 128)
        getitem_12 = None
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
        query = query_states_1.contiguous()
        query_states_1 = None
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
        add_4 = variance_1 + 1e-05
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
        add_6 = variance_2 + 1e-05
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
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_7.view((1, 2, -1, 128))
        linear_7 = None
        query_states_2 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_5 = linear_8.view((1, 2, -1, 128))
        linear_8 = None
        key_states_2 = view_5.transpose(1, 2)
        view_5 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_6 = linear_9.view((1, 2, -1, 128))
        linear_9 = None
        value_states_1 = view_6.transpose(1, 2)
        view_6 = None
        cos_4 = cos_1.unsqueeze(1)
        sin_4 = sin_1.unsqueeze(1)
        getitem_14 = cos_4[(Ellipsis, slice(None, 64, None))]
        cos_4 = None
        cos_5 = getitem_14.repeat_interleave(2, dim=-1)
        getitem_14 = None
        getitem_15 = sin_4[(Ellipsis, slice(None, 64, None))]
        sin_4 = None
        sin_5 = getitem_15.repeat_interleave(2, dim=-1)
        getitem_15 = None
        float_9 = query_states_2.float()
        mul_14 = float_9 * cos_5
        float_9 = None
        x1_2 = query_states_2[(Ellipsis, slice(0, None, 2))]
        x2_2 = query_states_2[(Ellipsis, slice(1, None, 2))]
        query_states_2 = None
        neg_2 = -x2_2
        x2_2 = None
        stack_2 = torch.stack((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        flatten_2 = stack_2.flatten(-2)
        stack_2 = None
        float_10 = flatten_2.float()
        flatten_2 = None
        mul_15 = float_10 * sin_5
        float_10 = None
        q_embed_1 = mul_14 + mul_15
        mul_14 = mul_15 = None
        float_11 = key_states_2.float()
        mul_16 = float_11 * cos_5
        float_11 = cos_5 = None
        x1_3 = key_states_2[(Ellipsis, slice(0, None, 2))]
        x2_3 = key_states_2[(Ellipsis, slice(1, None, 2))]
        key_states_2 = None
        neg_3 = -x2_3
        x2_3 = None
        stack_3 = torch.stack((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        flatten_3 = stack_3.flatten(-2)
        stack_3 = None
        float_12 = flatten_3.float()
        flatten_3 = None
        mul_17 = float_12 * sin_5
        float_12 = sin_5 = None
        k_embed_1 = mul_16 + mul_17
        mul_16 = mul_17 = None
        query_states_3 = q_embed_1.to(torch.bfloat16)
        q_embed_1 = None
        key_states_3 = k_embed_1.to(torch.bfloat16)
        k_embed_1 = None
        getitem_20 = key_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_13 = getitem_20.expand(1, 2, 8, 2, 128)
        getitem_20 = None
        key_2 = hidden_states_13.reshape(1, 16, 2, 128)
        hidden_states_13 = None
        getitem_21 = value_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_14 = getitem_21.expand(1, 2, 8, 2, 128)
        getitem_21 = None
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
        query_1 = query_states_3.contiguous()
        query_states_3 = None
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
        add_10 = variance_3 + 1e-05
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_17 = hidden_states_16 * rsqrt_3
        hidden_states_16 = rsqrt_3 = None
        to_13 = hidden_states_17.to(torch.bfloat16)
        hidden_states_17 = None
        hidden_states_18 = (
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_13
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
        add_12 = variance_4 + 1e-05
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_21 = hidden_states_20 * rsqrt_4
        hidden_states_20 = rsqrt_4 = None
        to_15 = hidden_states_21.to(torch.bfloat16)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_14.view((1, 2, -1, 128))
        linear_14 = None
        query_states_4 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_8 = linear_15.view((1, 2, -1, 128))
        linear_15 = None
        key_states_4 = view_8.transpose(1, 2)
        view_8 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_9 = linear_16.view((1, 2, -1, 128))
        linear_16 = None
        value_states_2 = view_9.transpose(1, 2)
        view_9 = None
        cos_6 = cos_1.unsqueeze(1)
        sin_6 = sin_1.unsqueeze(1)
        getitem_23 = cos_6[(Ellipsis, slice(None, 64, None))]
        cos_6 = None
        cos_7 = getitem_23.repeat_interleave(2, dim=-1)
        getitem_23 = None
        getitem_24 = sin_6[(Ellipsis, slice(None, 64, None))]
        sin_6 = None
        sin_7 = getitem_24.repeat_interleave(2, dim=-1)
        getitem_24 = None
        float_13 = query_states_4.float()
        mul_23 = float_13 * cos_7
        float_13 = None
        x1_4 = query_states_4[(Ellipsis, slice(0, None, 2))]
        x2_4 = query_states_4[(Ellipsis, slice(1, None, 2))]
        query_states_4 = None
        neg_4 = -x2_4
        x2_4 = None
        stack_4 = torch.stack((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        flatten_4 = stack_4.flatten(-2)
        stack_4 = None
        float_14 = flatten_4.float()
        flatten_4 = None
        mul_24 = float_14 * sin_7
        float_14 = None
        q_embed_2 = mul_23 + mul_24
        mul_23 = mul_24 = None
        float_15 = key_states_4.float()
        mul_25 = float_15 * cos_7
        float_15 = cos_7 = None
        x1_5 = key_states_4[(Ellipsis, slice(0, None, 2))]
        x2_5 = key_states_4[(Ellipsis, slice(1, None, 2))]
        key_states_4 = None
        neg_5 = -x2_5
        x2_5 = None
        stack_5 = torch.stack((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        flatten_5 = stack_5.flatten(-2)
        stack_5 = None
        float_16 = flatten_5.float()
        flatten_5 = None
        mul_26 = float_16 * sin_7
        float_16 = sin_7 = None
        k_embed_2 = mul_25 + mul_26
        mul_25 = mul_26 = None
        query_states_5 = q_embed_2.to(torch.bfloat16)
        q_embed_2 = None
        key_states_5 = k_embed_2.to(torch.bfloat16)
        k_embed_2 = None
        getitem_29 = key_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_23 = getitem_29.expand(1, 2, 8, 2, 128)
        getitem_29 = None
        key_4 = hidden_states_23.reshape(1, 16, 2, 128)
        hidden_states_23 = None
        getitem_30 = value_states_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_24 = getitem_30.expand(1, 2, 8, 2, 128)
        getitem_30 = None
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
        query_2 = query_states_5.contiguous()
        query_states_5 = None
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
        add_16 = variance_5 + 1e-05
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_27 = hidden_states_26 * rsqrt_5
        hidden_states_26 = rsqrt_5 = None
        to_19 = hidden_states_27.to(torch.bfloat16)
        hidden_states_27 = None
        hidden_states_28 = (
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_19
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
        add_18 = variance_6 + 1e-05
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_31 = hidden_states_30 * rsqrt_6
        hidden_states_30 = rsqrt_6 = None
        to_21 = hidden_states_31.to(torch.bfloat16)
        hidden_states_31 = None
        hidden_states_32 = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_21
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_21.view((1, 2, -1, 128))
        linear_21 = None
        query_states_6 = view_10.transpose(1, 2)
        view_10 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_11 = linear_22.view((1, 2, -1, 128))
        linear_22 = None
        key_states_6 = view_11.transpose(1, 2)
        view_11 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_12 = linear_23.view((1, 2, -1, 128))
        linear_23 = None
        value_states_3 = view_12.transpose(1, 2)
        view_12 = None
        cos_8 = cos_1.unsqueeze(1)
        sin_8 = sin_1.unsqueeze(1)
        getitem_32 = cos_8[(Ellipsis, slice(None, 64, None))]
        cos_8 = None
        cos_9 = getitem_32.repeat_interleave(2, dim=-1)
        getitem_32 = None
        getitem_33 = sin_8[(Ellipsis, slice(None, 64, None))]
        sin_8 = None
        sin_9 = getitem_33.repeat_interleave(2, dim=-1)
        getitem_33 = None
        float_17 = query_states_6.float()
        mul_32 = float_17 * cos_9
        float_17 = None
        x1_6 = query_states_6[(Ellipsis, slice(0, None, 2))]
        x2_6 = query_states_6[(Ellipsis, slice(1, None, 2))]
        query_states_6 = None
        neg_6 = -x2_6
        x2_6 = None
        stack_6 = torch.stack((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        flatten_6 = stack_6.flatten(-2)
        stack_6 = None
        float_18 = flatten_6.float()
        flatten_6 = None
        mul_33 = float_18 * sin_9
        float_18 = None
        q_embed_3 = mul_32 + mul_33
        mul_32 = mul_33 = None
        float_19 = key_states_6.float()
        mul_34 = float_19 * cos_9
        float_19 = cos_9 = None
        x1_7 = key_states_6[(Ellipsis, slice(0, None, 2))]
        x2_7 = key_states_6[(Ellipsis, slice(1, None, 2))]
        key_states_6 = None
        neg_7 = -x2_7
        x2_7 = None
        stack_7 = torch.stack((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        flatten_7 = stack_7.flatten(-2)
        stack_7 = None
        float_20 = flatten_7.float()
        flatten_7 = None
        mul_35 = float_20 * sin_9
        float_20 = sin_9 = None
        k_embed_3 = mul_34 + mul_35
        mul_34 = mul_35 = None
        query_states_7 = q_embed_3.to(torch.bfloat16)
        q_embed_3 = None
        key_states_7 = k_embed_3.to(torch.bfloat16)
        k_embed_3 = None
        getitem_38 = key_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_33 = getitem_38.expand(1, 2, 8, 2, 128)
        getitem_38 = None
        key_6 = hidden_states_33.reshape(1, 16, 2, 128)
        hidden_states_33 = None
        getitem_39 = value_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_34 = getitem_39.expand(1, 2, 8, 2, 128)
        getitem_39 = None
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
        query_3 = query_states_7.contiguous()
        query_states_7 = None
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
        add_22 = variance_7 + 1e-05
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_37 = hidden_states_36 * rsqrt_7
        hidden_states_36 = rsqrt_7 = None
        to_25 = hidden_states_37.to(torch.bfloat16)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_25
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
        add_24 = variance_8 + 1e-05
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_41 = hidden_states_40 * rsqrt_8
        hidden_states_40 = rsqrt_8 = None
        to_27 = hidden_states_41.to(torch.bfloat16)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_27
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_28.view((1, 2, -1, 128))
        linear_28 = None
        query_states_8 = view_13.transpose(1, 2)
        view_13 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_14 = linear_29.view((1, 2, -1, 128))
        linear_29 = None
        key_states_8 = view_14.transpose(1, 2)
        view_14 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_15 = linear_30.view((1, 2, -1, 128))
        linear_30 = None
        value_states_4 = view_15.transpose(1, 2)
        view_15 = None
        cos_10 = cos_1.unsqueeze(1)
        sin_10 = sin_1.unsqueeze(1)
        getitem_41 = cos_10[(Ellipsis, slice(None, 64, None))]
        cos_10 = None
        cos_11 = getitem_41.repeat_interleave(2, dim=-1)
        getitem_41 = None
        getitem_42 = sin_10[(Ellipsis, slice(None, 64, None))]
        sin_10 = None
        sin_11 = getitem_42.repeat_interleave(2, dim=-1)
        getitem_42 = None
        float_21 = query_states_8.float()
        mul_41 = float_21 * cos_11
        float_21 = None
        x1_8 = query_states_8[(Ellipsis, slice(0, None, 2))]
        x2_8 = query_states_8[(Ellipsis, slice(1, None, 2))]
        query_states_8 = None
        neg_8 = -x2_8
        x2_8 = None
        stack_8 = torch.stack((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        flatten_8 = stack_8.flatten(-2)
        stack_8 = None
        float_22 = flatten_8.float()
        flatten_8 = None
        mul_42 = float_22 * sin_11
        float_22 = None
        q_embed_4 = mul_41 + mul_42
        mul_41 = mul_42 = None
        float_23 = key_states_8.float()
        mul_43 = float_23 * cos_11
        float_23 = cos_11 = None
        x1_9 = key_states_8[(Ellipsis, slice(0, None, 2))]
        x2_9 = key_states_8[(Ellipsis, slice(1, None, 2))]
        key_states_8 = None
        neg_9 = -x2_9
        x2_9 = None
        stack_9 = torch.stack((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        flatten_9 = stack_9.flatten(-2)
        stack_9 = None
        float_24 = flatten_9.float()
        flatten_9 = None
        mul_44 = float_24 * sin_11
        float_24 = sin_11 = None
        k_embed_4 = mul_43 + mul_44
        mul_43 = mul_44 = None
        query_states_9 = q_embed_4.to(torch.bfloat16)
        q_embed_4 = None
        key_states_9 = k_embed_4.to(torch.bfloat16)
        k_embed_4 = None
        getitem_47 = key_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_43 = getitem_47.expand(1, 2, 8, 2, 128)
        getitem_47 = None
        key_8 = hidden_states_43.reshape(1, 16, 2, 128)
        hidden_states_43 = None
        getitem_48 = value_states_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_44 = getitem_48.expand(1, 2, 8, 2, 128)
        getitem_48 = None
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
        query_4 = query_states_9.contiguous()
        query_states_9 = None
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
        add_28 = variance_9 + 1e-05
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_47 = hidden_states_46 * rsqrt_9
        hidden_states_46 = rsqrt_9 = None
        to_31 = hidden_states_47.to(torch.bfloat16)
        hidden_states_47 = None
        hidden_states_48 = (
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_31
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
        add_30 = variance_10 + 1e-05
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_51 = hidden_states_50 * rsqrt_10
        hidden_states_50 = rsqrt_10 = None
        to_33 = hidden_states_51.to(torch.bfloat16)
        hidden_states_51 = None
        hidden_states_52 = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_33
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_35.view((1, 2, -1, 128))
        linear_35 = None
        query_states_10 = view_16.transpose(1, 2)
        view_16 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_17 = linear_36.view((1, 2, -1, 128))
        linear_36 = None
        key_states_10 = view_17.transpose(1, 2)
        view_17 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_18 = linear_37.view((1, 2, -1, 128))
        linear_37 = None
        value_states_5 = view_18.transpose(1, 2)
        view_18 = None
        cos_12 = cos_1.unsqueeze(1)
        sin_12 = sin_1.unsqueeze(1)
        getitem_50 = cos_12[(Ellipsis, slice(None, 64, None))]
        cos_12 = None
        cos_13 = getitem_50.repeat_interleave(2, dim=-1)
        getitem_50 = None
        getitem_51 = sin_12[(Ellipsis, slice(None, 64, None))]
        sin_12 = None
        sin_13 = getitem_51.repeat_interleave(2, dim=-1)
        getitem_51 = None
        float_25 = query_states_10.float()
        mul_50 = float_25 * cos_13
        float_25 = None
        x1_10 = query_states_10[(Ellipsis, slice(0, None, 2))]
        x2_10 = query_states_10[(Ellipsis, slice(1, None, 2))]
        query_states_10 = None
        neg_10 = -x2_10
        x2_10 = None
        stack_10 = torch.stack((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        flatten_10 = stack_10.flatten(-2)
        stack_10 = None
        float_26 = flatten_10.float()
        flatten_10 = None
        mul_51 = float_26 * sin_13
        float_26 = None
        q_embed_5 = mul_50 + mul_51
        mul_50 = mul_51 = None
        float_27 = key_states_10.float()
        mul_52 = float_27 * cos_13
        float_27 = cos_13 = None
        x1_11 = key_states_10[(Ellipsis, slice(0, None, 2))]
        x2_11 = key_states_10[(Ellipsis, slice(1, None, 2))]
        key_states_10 = None
        neg_11 = -x2_11
        x2_11 = None
        stack_11 = torch.stack((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        flatten_11 = stack_11.flatten(-2)
        stack_11 = None
        float_28 = flatten_11.float()
        flatten_11 = None
        mul_53 = float_28 * sin_13
        float_28 = sin_13 = None
        k_embed_5 = mul_52 + mul_53
        mul_52 = mul_53 = None
        query_states_11 = q_embed_5.to(torch.bfloat16)
        q_embed_5 = None
        key_states_11 = k_embed_5.to(torch.bfloat16)
        k_embed_5 = None
        getitem_56 = key_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_53 = getitem_56.expand(1, 2, 8, 2, 128)
        getitem_56 = None
        key_10 = hidden_states_53.reshape(1, 16, 2, 128)
        hidden_states_53 = None
        getitem_57 = value_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_54 = getitem_57.expand(1, 2, 8, 2, 128)
        getitem_57 = None
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
        query_5 = query_states_11.contiguous()
        query_states_11 = None
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
        add_34 = variance_11 + 1e-05
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_57 = hidden_states_56 * rsqrt_11
        hidden_states_56 = rsqrt_11 = None
        to_37 = hidden_states_57.to(torch.bfloat16)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_37
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
        add_36 = variance_12 + 1e-05
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_61 = hidden_states_60 * rsqrt_12
        hidden_states_60 = rsqrt_12 = None
        to_39 = hidden_states_61.to(torch.bfloat16)
        hidden_states_61 = None
        hidden_states_62 = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_39
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_42.view((1, 2, -1, 128))
        linear_42 = None
        query_states_12 = view_19.transpose(1, 2)
        view_19 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_20 = linear_43.view((1, 2, -1, 128))
        linear_43 = None
        key_states_12 = view_20.transpose(1, 2)
        view_20 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_62 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_21 = linear_44.view((1, 2, -1, 128))
        linear_44 = None
        value_states_6 = view_21.transpose(1, 2)
        view_21 = None
        cos_14 = cos_1.unsqueeze(1)
        sin_14 = sin_1.unsqueeze(1)
        getitem_59 = cos_14[(Ellipsis, slice(None, 64, None))]
        cos_14 = None
        cos_15 = getitem_59.repeat_interleave(2, dim=-1)
        getitem_59 = None
        getitem_60 = sin_14[(Ellipsis, slice(None, 64, None))]
        sin_14 = None
        sin_15 = getitem_60.repeat_interleave(2, dim=-1)
        getitem_60 = None
        float_29 = query_states_12.float()
        mul_59 = float_29 * cos_15
        float_29 = None
        x1_12 = query_states_12[(Ellipsis, slice(0, None, 2))]
        x2_12 = query_states_12[(Ellipsis, slice(1, None, 2))]
        query_states_12 = None
        neg_12 = -x2_12
        x2_12 = None
        stack_12 = torch.stack((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        flatten_12 = stack_12.flatten(-2)
        stack_12 = None
        float_30 = flatten_12.float()
        flatten_12 = None
        mul_60 = float_30 * sin_15
        float_30 = None
        q_embed_6 = mul_59 + mul_60
        mul_59 = mul_60 = None
        float_31 = key_states_12.float()
        mul_61 = float_31 * cos_15
        float_31 = cos_15 = None
        x1_13 = key_states_12[(Ellipsis, slice(0, None, 2))]
        x2_13 = key_states_12[(Ellipsis, slice(1, None, 2))]
        key_states_12 = None
        neg_13 = -x2_13
        x2_13 = None
        stack_13 = torch.stack((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        flatten_13 = stack_13.flatten(-2)
        stack_13 = None
        float_32 = flatten_13.float()
        flatten_13 = None
        mul_62 = float_32 * sin_15
        float_32 = sin_15 = None
        k_embed_6 = mul_61 + mul_62
        mul_61 = mul_62 = None
        query_states_13 = q_embed_6.to(torch.bfloat16)
        q_embed_6 = None
        key_states_13 = k_embed_6.to(torch.bfloat16)
        k_embed_6 = None
        getitem_65 = key_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_63 = getitem_65.expand(1, 2, 8, 2, 128)
        getitem_65 = None
        key_12 = hidden_states_63.reshape(1, 16, 2, 128)
        hidden_states_63 = None
        getitem_66 = value_states_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_64 = getitem_66.expand(1, 2, 8, 2, 128)
        getitem_66 = None
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
        query_6 = query_states_13.contiguous()
        query_states_13 = None
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
        add_40 = variance_13 + 1e-05
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_67 = hidden_states_66 * rsqrt_13
        hidden_states_66 = rsqrt_13 = None
        to_43 = hidden_states_67.to(torch.bfloat16)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_43
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
        add_42 = variance_14 + 1e-05
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_71 = hidden_states_70 * rsqrt_14
        hidden_states_70 = rsqrt_14 = None
        to_45 = hidden_states_71.to(torch.bfloat16)
        hidden_states_71 = None
        hidden_states_72 = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_45
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_49.view((1, 2, -1, 128))
        linear_49 = None
        query_states_14 = view_22.transpose(1, 2)
        view_22 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_23 = linear_50.view((1, 2, -1, 128))
        linear_50 = None
        key_states_14 = view_23.transpose(1, 2)
        view_23 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_24 = linear_51.view((1, 2, -1, 128))
        linear_51 = None
        value_states_7 = view_24.transpose(1, 2)
        view_24 = None
        cos_16 = cos_1.unsqueeze(1)
        sin_16 = sin_1.unsqueeze(1)
        getitem_68 = cos_16[(Ellipsis, slice(None, 64, None))]
        cos_16 = None
        cos_17 = getitem_68.repeat_interleave(2, dim=-1)
        getitem_68 = None
        getitem_69 = sin_16[(Ellipsis, slice(None, 64, None))]
        sin_16 = None
        sin_17 = getitem_69.repeat_interleave(2, dim=-1)
        getitem_69 = None
        float_33 = query_states_14.float()
        mul_68 = float_33 * cos_17
        float_33 = None
        x1_14 = query_states_14[(Ellipsis, slice(0, None, 2))]
        x2_14 = query_states_14[(Ellipsis, slice(1, None, 2))]
        query_states_14 = None
        neg_14 = -x2_14
        x2_14 = None
        stack_14 = torch.stack((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        flatten_14 = stack_14.flatten(-2)
        stack_14 = None
        float_34 = flatten_14.float()
        flatten_14 = None
        mul_69 = float_34 * sin_17
        float_34 = None
        q_embed_7 = mul_68 + mul_69
        mul_68 = mul_69 = None
        float_35 = key_states_14.float()
        mul_70 = float_35 * cos_17
        float_35 = cos_17 = None
        x1_15 = key_states_14[(Ellipsis, slice(0, None, 2))]
        x2_15 = key_states_14[(Ellipsis, slice(1, None, 2))]
        key_states_14 = None
        neg_15 = -x2_15
        x2_15 = None
        stack_15 = torch.stack((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        flatten_15 = stack_15.flatten(-2)
        stack_15 = None
        float_36 = flatten_15.float()
        flatten_15 = None
        mul_71 = float_36 * sin_17
        float_36 = sin_17 = None
        k_embed_7 = mul_70 + mul_71
        mul_70 = mul_71 = None
        query_states_15 = q_embed_7.to(torch.bfloat16)
        q_embed_7 = None
        key_states_15 = k_embed_7.to(torch.bfloat16)
        k_embed_7 = None
        getitem_74 = key_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_73 = getitem_74.expand(1, 2, 8, 2, 128)
        getitem_74 = None
        key_14 = hidden_states_73.reshape(1, 16, 2, 128)
        hidden_states_73 = None
        getitem_75 = value_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_74 = getitem_75.expand(1, 2, 8, 2, 128)
        getitem_75 = None
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
        query_7 = query_states_15.contiguous()
        query_states_15 = None
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
        add_46 = variance_15 + 1e-05
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_77 = hidden_states_76 * rsqrt_15
        hidden_states_76 = rsqrt_15 = None
        to_49 = hidden_states_77.to(torch.bfloat16)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_49
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
        add_48 = variance_16 + 1e-05
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_81 = hidden_states_80 * rsqrt_16
        hidden_states_80 = rsqrt_16 = None
        to_51 = hidden_states_81.to(torch.bfloat16)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_51
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_56.view((1, 2, -1, 128))
        linear_56 = None
        query_states_16 = view_25.transpose(1, 2)
        view_25 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_26 = linear_57.view((1, 2, -1, 128))
        linear_57 = None
        key_states_16 = view_26.transpose(1, 2)
        view_26 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_82 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_27 = linear_58.view((1, 2, -1, 128))
        linear_58 = None
        value_states_8 = view_27.transpose(1, 2)
        view_27 = None
        cos_18 = cos_1.unsqueeze(1)
        sin_18 = sin_1.unsqueeze(1)
        getitem_77 = cos_18[(Ellipsis, slice(None, 64, None))]
        cos_18 = None
        cos_19 = getitem_77.repeat_interleave(2, dim=-1)
        getitem_77 = None
        getitem_78 = sin_18[(Ellipsis, slice(None, 64, None))]
        sin_18 = None
        sin_19 = getitem_78.repeat_interleave(2, dim=-1)
        getitem_78 = None
        float_37 = query_states_16.float()
        mul_77 = float_37 * cos_19
        float_37 = None
        x1_16 = query_states_16[(Ellipsis, slice(0, None, 2))]
        x2_16 = query_states_16[(Ellipsis, slice(1, None, 2))]
        query_states_16 = None
        neg_16 = -x2_16
        x2_16 = None
        stack_16 = torch.stack((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        flatten_16 = stack_16.flatten(-2)
        stack_16 = None
        float_38 = flatten_16.float()
        flatten_16 = None
        mul_78 = float_38 * sin_19
        float_38 = None
        q_embed_8 = mul_77 + mul_78
        mul_77 = mul_78 = None
        float_39 = key_states_16.float()
        mul_79 = float_39 * cos_19
        float_39 = cos_19 = None
        x1_17 = key_states_16[(Ellipsis, slice(0, None, 2))]
        x2_17 = key_states_16[(Ellipsis, slice(1, None, 2))]
        key_states_16 = None
        neg_17 = -x2_17
        x2_17 = None
        stack_17 = torch.stack((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        flatten_17 = stack_17.flatten(-2)
        stack_17 = None
        float_40 = flatten_17.float()
        flatten_17 = None
        mul_80 = float_40 * sin_19
        float_40 = sin_19 = None
        k_embed_8 = mul_79 + mul_80
        mul_79 = mul_80 = None
        query_states_17 = q_embed_8.to(torch.bfloat16)
        q_embed_8 = None
        key_states_17 = k_embed_8.to(torch.bfloat16)
        k_embed_8 = None
        getitem_83 = key_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_83 = getitem_83.expand(1, 2, 8, 2, 128)
        getitem_83 = None
        key_16 = hidden_states_83.reshape(1, 16, 2, 128)
        hidden_states_83 = None
        getitem_84 = value_states_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_84 = getitem_84.expand(1, 2, 8, 2, 128)
        getitem_84 = None
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
        query_8 = query_states_17.contiguous()
        query_states_17 = None
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
        add_52 = variance_17 + 1e-05
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_87 = hidden_states_86 * rsqrt_17
        hidden_states_86 = rsqrt_17 = None
        to_55 = hidden_states_87.to(torch.bfloat16)
        hidden_states_87 = None
        hidden_states_88 = (
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_55
        )
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_55
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
        add_54 = variance_18 + 1e-05
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_91 = hidden_states_90 * rsqrt_18
        hidden_states_90 = rsqrt_18 = None
        to_57 = hidden_states_91.to(torch.bfloat16)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_57
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_57
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_63.view((1, 2, -1, 128))
        linear_63 = None
        query_states_18 = view_28.transpose(1, 2)
        view_28 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_29 = linear_64.view((1, 2, -1, 128))
        linear_64 = None
        key_states_18 = view_29.transpose(1, 2)
        view_29 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_92 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_30 = linear_65.view((1, 2, -1, 128))
        linear_65 = None
        value_states_9 = view_30.transpose(1, 2)
        view_30 = None
        cos_20 = cos_1.unsqueeze(1)
        sin_20 = sin_1.unsqueeze(1)
        getitem_86 = cos_20[(Ellipsis, slice(None, 64, None))]
        cos_20 = None
        cos_21 = getitem_86.repeat_interleave(2, dim=-1)
        getitem_86 = None
        getitem_87 = sin_20[(Ellipsis, slice(None, 64, None))]
        sin_20 = None
        sin_21 = getitem_87.repeat_interleave(2, dim=-1)
        getitem_87 = None
        float_41 = query_states_18.float()
        mul_86 = float_41 * cos_21
        float_41 = None
        x1_18 = query_states_18[(Ellipsis, slice(0, None, 2))]
        x2_18 = query_states_18[(Ellipsis, slice(1, None, 2))]
        query_states_18 = None
        neg_18 = -x2_18
        x2_18 = None
        stack_18 = torch.stack((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        flatten_18 = stack_18.flatten(-2)
        stack_18 = None
        float_42 = flatten_18.float()
        flatten_18 = None
        mul_87 = float_42 * sin_21
        float_42 = None
        q_embed_9 = mul_86 + mul_87
        mul_86 = mul_87 = None
        float_43 = key_states_18.float()
        mul_88 = float_43 * cos_21
        float_43 = cos_21 = None
        x1_19 = key_states_18[(Ellipsis, slice(0, None, 2))]
        x2_19 = key_states_18[(Ellipsis, slice(1, None, 2))]
        key_states_18 = None
        neg_19 = -x2_19
        x2_19 = None
        stack_19 = torch.stack((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        flatten_19 = stack_19.flatten(-2)
        stack_19 = None
        float_44 = flatten_19.float()
        flatten_19 = None
        mul_89 = float_44 * sin_21
        float_44 = sin_21 = None
        k_embed_9 = mul_88 + mul_89
        mul_88 = mul_89 = None
        query_states_19 = q_embed_9.to(torch.bfloat16)
        q_embed_9 = None
        key_states_19 = k_embed_9.to(torch.bfloat16)
        k_embed_9 = None
        getitem_92 = key_states_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_93 = getitem_92.expand(1, 2, 8, 2, 128)
        getitem_92 = None
        key_18 = hidden_states_93.reshape(1, 16, 2, 128)
        hidden_states_93 = None
        getitem_93 = value_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_94 = getitem_93.expand(1, 2, 8, 2, 128)
        getitem_93 = None
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
        query_9 = query_states_19.contiguous()
        query_states_19 = None
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
        add_58 = variance_19 + 1e-05
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_58)
        add_58 = None
        hidden_states_97 = hidden_states_96 * rsqrt_19
        hidden_states_96 = rsqrt_19 = None
        to_61 = hidden_states_97.to(torch.bfloat16)
        hidden_states_97 = None
        hidden_states_98 = (
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_61
        )
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_61
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
        add_60 = variance_20 + 1e-05
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_60)
        add_60 = None
        hidden_states_101 = hidden_states_100 * rsqrt_20
        hidden_states_100 = rsqrt_20 = None
        to_63 = hidden_states_101.to(torch.bfloat16)
        hidden_states_101 = None
        hidden_states_102 = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_63
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_63
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_70.view((1, 2, -1, 128))
        linear_70 = None
        query_states_20 = view_31.transpose(1, 2)
        view_31 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_32 = linear_71.view((1, 2, -1, 128))
        linear_71 = None
        key_states_20 = view_32.transpose(1, 2)
        view_32 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_102 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_33 = linear_72.view((1, 2, -1, 128))
        linear_72 = None
        value_states_10 = view_33.transpose(1, 2)
        view_33 = None
        cos_22 = cos_1.unsqueeze(1)
        sin_22 = sin_1.unsqueeze(1)
        getitem_95 = cos_22[(Ellipsis, slice(None, 64, None))]
        cos_22 = None
        cos_23 = getitem_95.repeat_interleave(2, dim=-1)
        getitem_95 = None
        getitem_96 = sin_22[(Ellipsis, slice(None, 64, None))]
        sin_22 = None
        sin_23 = getitem_96.repeat_interleave(2, dim=-1)
        getitem_96 = None
        float_45 = query_states_20.float()
        mul_95 = float_45 * cos_23
        float_45 = None
        x1_20 = query_states_20[(Ellipsis, slice(0, None, 2))]
        x2_20 = query_states_20[(Ellipsis, slice(1, None, 2))]
        query_states_20 = None
        neg_20 = -x2_20
        x2_20 = None
        stack_20 = torch.stack((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        flatten_20 = stack_20.flatten(-2)
        stack_20 = None
        float_46 = flatten_20.float()
        flatten_20 = None
        mul_96 = float_46 * sin_23
        float_46 = None
        q_embed_10 = mul_95 + mul_96
        mul_95 = mul_96 = None
        float_47 = key_states_20.float()
        mul_97 = float_47 * cos_23
        float_47 = cos_23 = None
        x1_21 = key_states_20[(Ellipsis, slice(0, None, 2))]
        x2_21 = key_states_20[(Ellipsis, slice(1, None, 2))]
        key_states_20 = None
        neg_21 = -x2_21
        x2_21 = None
        stack_21 = torch.stack((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        flatten_21 = stack_21.flatten(-2)
        stack_21 = None
        float_48 = flatten_21.float()
        flatten_21 = None
        mul_98 = float_48 * sin_23
        float_48 = sin_23 = None
        k_embed_10 = mul_97 + mul_98
        mul_97 = mul_98 = None
        query_states_21 = q_embed_10.to(torch.bfloat16)
        q_embed_10 = None
        key_states_21 = k_embed_10.to(torch.bfloat16)
        k_embed_10 = None
        getitem_101 = key_states_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_103 = getitem_101.expand(1, 2, 8, 2, 128)
        getitem_101 = None
        key_20 = hidden_states_103.reshape(1, 16, 2, 128)
        hidden_states_103 = None
        getitem_102 = value_states_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_104 = getitem_102.expand(1, 2, 8, 2, 128)
        getitem_102 = None
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
        query_10 = query_states_21.contiguous()
        query_states_21 = None
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
        add_64 = variance_21 + 1e-05
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_107 = hidden_states_106 * rsqrt_21
        hidden_states_106 = rsqrt_21 = None
        to_67 = hidden_states_107.to(torch.bfloat16)
        hidden_states_107 = None
        hidden_states_108 = (
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_67
        )
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_67
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
        add_66 = variance_22 + 1e-05
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_111 = hidden_states_110 * rsqrt_22
        hidden_states_110 = rsqrt_22 = None
        to_69 = hidden_states_111.to(torch.bfloat16)
        hidden_states_111 = None
        hidden_states_112 = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_69
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_69
        ) = None
        linear_77 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_77.view((1, 2, -1, 128))
        linear_77 = None
        query_states_22 = view_34.transpose(1, 2)
        view_34 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_35 = linear_78.view((1, 2, -1, 128))
        linear_78 = None
        key_states_22 = view_35.transpose(1, 2)
        view_35 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_112 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_36 = linear_79.view((1, 2, -1, 128))
        linear_79 = None
        value_states_11 = view_36.transpose(1, 2)
        view_36 = None
        cos_24 = cos_1.unsqueeze(1)
        sin_24 = sin_1.unsqueeze(1)
        getitem_104 = cos_24[(Ellipsis, slice(None, 64, None))]
        cos_24 = None
        cos_25 = getitem_104.repeat_interleave(2, dim=-1)
        getitem_104 = None
        getitem_105 = sin_24[(Ellipsis, slice(None, 64, None))]
        sin_24 = None
        sin_25 = getitem_105.repeat_interleave(2, dim=-1)
        getitem_105 = None
        float_49 = query_states_22.float()
        mul_104 = float_49 * cos_25
        float_49 = None
        x1_22 = query_states_22[(Ellipsis, slice(0, None, 2))]
        x2_22 = query_states_22[(Ellipsis, slice(1, None, 2))]
        query_states_22 = None
        neg_22 = -x2_22
        x2_22 = None
        stack_22 = torch.stack((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        flatten_22 = stack_22.flatten(-2)
        stack_22 = None
        float_50 = flatten_22.float()
        flatten_22 = None
        mul_105 = float_50 * sin_25
        float_50 = None
        q_embed_11 = mul_104 + mul_105
        mul_104 = mul_105 = None
        float_51 = key_states_22.float()
        mul_106 = float_51 * cos_25
        float_51 = cos_25 = None
        x1_23 = key_states_22[(Ellipsis, slice(0, None, 2))]
        x2_23 = key_states_22[(Ellipsis, slice(1, None, 2))]
        key_states_22 = None
        neg_23 = -x2_23
        x2_23 = None
        stack_23 = torch.stack((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        flatten_23 = stack_23.flatten(-2)
        stack_23 = None
        float_52 = flatten_23.float()
        flatten_23 = None
        mul_107 = float_52 * sin_25
        float_52 = sin_25 = None
        k_embed_11 = mul_106 + mul_107
        mul_106 = mul_107 = None
        query_states_23 = q_embed_11.to(torch.bfloat16)
        q_embed_11 = None
        key_states_23 = k_embed_11.to(torch.bfloat16)
        k_embed_11 = None
        getitem_110 = key_states_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_113 = getitem_110.expand(1, 2, 8, 2, 128)
        getitem_110 = None
        key_22 = hidden_states_113.reshape(1, 16, 2, 128)
        hidden_states_113 = None
        getitem_111 = value_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_114 = getitem_111.expand(1, 2, 8, 2, 128)
        getitem_111 = None
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
        query_11 = query_states_23.contiguous()
        query_states_23 = None
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
        add_70 = variance_23 + 1e-05
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_117 = hidden_states_116 * rsqrt_23
        hidden_states_116 = rsqrt_23 = None
        to_73 = hidden_states_117.to(torch.bfloat16)
        hidden_states_117 = None
        hidden_states_118 = (
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_73
        )
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_73
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
        add_72 = variance_24 + 1e-05
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_121 = hidden_states_120 * rsqrt_24
        hidden_states_120 = rsqrt_24 = None
        to_75 = hidden_states_121.to(torch.bfloat16)
        hidden_states_121 = None
        hidden_states_122 = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
            * to_75
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            to_75
        ) = None
        linear_84 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_84.view((1, 2, -1, 128))
        linear_84 = None
        query_states_24 = view_37.transpose(1, 2)
        view_37 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_38 = linear_85.view((1, 2, -1, 128))
        linear_85 = None
        key_states_24 = view_38.transpose(1, 2)
        view_38 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_122 = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_39 = linear_86.view((1, 2, -1, 128))
        linear_86 = None
        value_states_12 = view_39.transpose(1, 2)
        view_39 = None
        cos_26 = cos_1.unsqueeze(1)
        sin_26 = sin_1.unsqueeze(1)
        getitem_113 = cos_26[(Ellipsis, slice(None, 64, None))]
        cos_26 = None
        cos_27 = getitem_113.repeat_interleave(2, dim=-1)
        getitem_113 = None
        getitem_114 = sin_26[(Ellipsis, slice(None, 64, None))]
        sin_26 = None
        sin_27 = getitem_114.repeat_interleave(2, dim=-1)
        getitem_114 = None
        float_53 = query_states_24.float()
        mul_113 = float_53 * cos_27
        float_53 = None
        x1_24 = query_states_24[(Ellipsis, slice(0, None, 2))]
        x2_24 = query_states_24[(Ellipsis, slice(1, None, 2))]
        query_states_24 = None
        neg_24 = -x2_24
        x2_24 = None
        stack_24 = torch.stack((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        flatten_24 = stack_24.flatten(-2)
        stack_24 = None
        float_54 = flatten_24.float()
        flatten_24 = None
        mul_114 = float_54 * sin_27
        float_54 = None
        q_embed_12 = mul_113 + mul_114
        mul_113 = mul_114 = None
        float_55 = key_states_24.float()
        mul_115 = float_55 * cos_27
        float_55 = cos_27 = None
        x1_25 = key_states_24[(Ellipsis, slice(0, None, 2))]
        x2_25 = key_states_24[(Ellipsis, slice(1, None, 2))]
        key_states_24 = None
        neg_25 = -x2_25
        x2_25 = None
        stack_25 = torch.stack((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        flatten_25 = stack_25.flatten(-2)
        stack_25 = None
        float_56 = flatten_25.float()
        flatten_25 = None
        mul_116 = float_56 * sin_27
        float_56 = sin_27 = None
        k_embed_12 = mul_115 + mul_116
        mul_115 = mul_116 = None
        query_states_25 = q_embed_12.to(torch.bfloat16)
        q_embed_12 = None
        key_states_25 = k_embed_12.to(torch.bfloat16)
        k_embed_12 = None
        getitem_119 = key_states_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_123 = getitem_119.expand(1, 2, 8, 2, 128)
        getitem_119 = None
        key_24 = hidden_states_123.reshape(1, 16, 2, 128)
        hidden_states_123 = None
        getitem_120 = value_states_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_124 = getitem_120.expand(1, 2, 8, 2, 128)
        getitem_120 = None
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
        query_12 = query_states_25.contiguous()
        query_states_25 = None
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
        add_76 = variance_25 + 1e-05
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_76)
        add_76 = None
        hidden_states_127 = hidden_states_126 * rsqrt_25
        hidden_states_126 = rsqrt_25 = None
        to_79 = hidden_states_127.to(torch.bfloat16)
        hidden_states_127 = None
        hidden_states_128 = (
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_79
        )
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_79
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
        add_78 = variance_26 + 1e-05
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_131 = hidden_states_130 * rsqrt_26
        hidden_states_130 = rsqrt_26 = None
        to_81 = hidden_states_131.to(torch.bfloat16)
        hidden_states_131 = None
        hidden_states_132 = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
            * to_81
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            to_81
        ) = None
        linear_91 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_91.view((1, 2, -1, 128))
        linear_91 = None
        query_states_26 = view_40.transpose(1, 2)
        view_40 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_41 = linear_92.view((1, 2, -1, 128))
        linear_92 = None
        key_states_26 = view_41.transpose(1, 2)
        view_41 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_132 = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_42 = linear_93.view((1, 2, -1, 128))
        linear_93 = None
        value_states_13 = view_42.transpose(1, 2)
        view_42 = None
        cos_28 = cos_1.unsqueeze(1)
        sin_28 = sin_1.unsqueeze(1)
        getitem_122 = cos_28[(Ellipsis, slice(None, 64, None))]
        cos_28 = None
        cos_29 = getitem_122.repeat_interleave(2, dim=-1)
        getitem_122 = None
        getitem_123 = sin_28[(Ellipsis, slice(None, 64, None))]
        sin_28 = None
        sin_29 = getitem_123.repeat_interleave(2, dim=-1)
        getitem_123 = None
        float_57 = query_states_26.float()
        mul_122 = float_57 * cos_29
        float_57 = None
        x1_26 = query_states_26[(Ellipsis, slice(0, None, 2))]
        x2_26 = query_states_26[(Ellipsis, slice(1, None, 2))]
        query_states_26 = None
        neg_26 = -x2_26
        x2_26 = None
        stack_26 = torch.stack((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        flatten_26 = stack_26.flatten(-2)
        stack_26 = None
        float_58 = flatten_26.float()
        flatten_26 = None
        mul_123 = float_58 * sin_29
        float_58 = None
        q_embed_13 = mul_122 + mul_123
        mul_122 = mul_123 = None
        float_59 = key_states_26.float()
        mul_124 = float_59 * cos_29
        float_59 = cos_29 = None
        x1_27 = key_states_26[(Ellipsis, slice(0, None, 2))]
        x2_27 = key_states_26[(Ellipsis, slice(1, None, 2))]
        key_states_26 = None
        neg_27 = -x2_27
        x2_27 = None
        stack_27 = torch.stack((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        flatten_27 = stack_27.flatten(-2)
        stack_27 = None
        float_60 = flatten_27.float()
        flatten_27 = None
        mul_125 = float_60 * sin_29
        float_60 = sin_29 = None
        k_embed_13 = mul_124 + mul_125
        mul_124 = mul_125 = None
        query_states_27 = q_embed_13.to(torch.bfloat16)
        q_embed_13 = None
        key_states_27 = k_embed_13.to(torch.bfloat16)
        k_embed_13 = None
        getitem_128 = key_states_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_133 = getitem_128.expand(1, 2, 8, 2, 128)
        getitem_128 = None
        key_26 = hidden_states_133.reshape(1, 16, 2, 128)
        hidden_states_133 = None
        getitem_129 = value_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_134 = getitem_129.expand(1, 2, 8, 2, 128)
        getitem_129 = None
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
        query_13 = query_states_27.contiguous()
        query_states_27 = None
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
        add_82 = variance_27 + 1e-05
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_137 = hidden_states_136 * rsqrt_27
        hidden_states_136 = rsqrt_27 = None
        to_85 = hidden_states_137.to(torch.bfloat16)
        hidden_states_137 = None
        hidden_states_138 = (
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_85
        )
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_85
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
        add_84 = variance_28 + 1e-05
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_84)
        add_84 = None
        hidden_states_141 = hidden_states_140 * rsqrt_28
        hidden_states_140 = rsqrt_28 = None
        to_87 = hidden_states_141.to(torch.bfloat16)
        hidden_states_141 = None
        hidden_states_142 = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
            * to_87
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            to_87
        ) = None
        linear_98 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_98.view((1, 2, -1, 128))
        linear_98 = None
        query_states_28 = view_43.transpose(1, 2)
        view_43 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_44 = linear_99.view((1, 2, -1, 128))
        linear_99 = None
        key_states_28 = view_44.transpose(1, 2)
        view_44 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_142 = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_45 = linear_100.view((1, 2, -1, 128))
        linear_100 = None
        value_states_14 = view_45.transpose(1, 2)
        view_45 = None
        cos_30 = cos_1.unsqueeze(1)
        sin_30 = sin_1.unsqueeze(1)
        getitem_131 = cos_30[(Ellipsis, slice(None, 64, None))]
        cos_30 = None
        cos_31 = getitem_131.repeat_interleave(2, dim=-1)
        getitem_131 = None
        getitem_132 = sin_30[(Ellipsis, slice(None, 64, None))]
        sin_30 = None
        sin_31 = getitem_132.repeat_interleave(2, dim=-1)
        getitem_132 = None
        float_61 = query_states_28.float()
        mul_131 = float_61 * cos_31
        float_61 = None
        x1_28 = query_states_28[(Ellipsis, slice(0, None, 2))]
        x2_28 = query_states_28[(Ellipsis, slice(1, None, 2))]
        query_states_28 = None
        neg_28 = -x2_28
        x2_28 = None
        stack_28 = torch.stack((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        flatten_28 = stack_28.flatten(-2)
        stack_28 = None
        float_62 = flatten_28.float()
        flatten_28 = None
        mul_132 = float_62 * sin_31
        float_62 = None
        q_embed_14 = mul_131 + mul_132
        mul_131 = mul_132 = None
        float_63 = key_states_28.float()
        mul_133 = float_63 * cos_31
        float_63 = cos_31 = None
        x1_29 = key_states_28[(Ellipsis, slice(0, None, 2))]
        x2_29 = key_states_28[(Ellipsis, slice(1, None, 2))]
        key_states_28 = None
        neg_29 = -x2_29
        x2_29 = None
        stack_29 = torch.stack((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        flatten_29 = stack_29.flatten(-2)
        stack_29 = None
        float_64 = flatten_29.float()
        flatten_29 = None
        mul_134 = float_64 * sin_31
        float_64 = sin_31 = None
        k_embed_14 = mul_133 + mul_134
        mul_133 = mul_134 = None
        query_states_29 = q_embed_14.to(torch.bfloat16)
        q_embed_14 = None
        key_states_29 = k_embed_14.to(torch.bfloat16)
        k_embed_14 = None
        getitem_137 = key_states_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_143 = getitem_137.expand(1, 2, 8, 2, 128)
        getitem_137 = None
        key_28 = hidden_states_143.reshape(1, 16, 2, 128)
        hidden_states_143 = None
        getitem_138 = value_states_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_144 = getitem_138.expand(1, 2, 8, 2, 128)
        getitem_138 = None
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
        query_14 = query_states_29.contiguous()
        query_states_29 = None
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
        add_88 = variance_29 + 1e-05
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_147 = hidden_states_146 * rsqrt_29
        hidden_states_146 = rsqrt_29 = None
        to_91 = hidden_states_147.to(torch.bfloat16)
        hidden_states_147 = None
        hidden_states_148 = (
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_91
        )
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_91
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
        add_90 = variance_30 + 1e-05
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_90)
        add_90 = None
        hidden_states_151 = hidden_states_150 * rsqrt_30
        hidden_states_150 = rsqrt_30 = None
        to_93 = hidden_states_151.to(torch.bfloat16)
        hidden_states_151 = None
        hidden_states_152 = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
            * to_93
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            to_93
        ) = None
        linear_105 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_105.view((1, 2, -1, 128))
        linear_105 = None
        query_states_30 = view_46.transpose(1, 2)
        view_46 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_47 = linear_106.view((1, 2, -1, 128))
        linear_106 = None
        key_states_30 = view_47.transpose(1, 2)
        view_47 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_152 = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_48 = linear_107.view((1, 2, -1, 128))
        linear_107 = None
        value_states_15 = view_48.transpose(1, 2)
        view_48 = None
        cos_32 = cos_1.unsqueeze(1)
        sin_32 = sin_1.unsqueeze(1)
        getitem_140 = cos_32[(Ellipsis, slice(None, 64, None))]
        cos_32 = None
        cos_33 = getitem_140.repeat_interleave(2, dim=-1)
        getitem_140 = None
        getitem_141 = sin_32[(Ellipsis, slice(None, 64, None))]
        sin_32 = None
        sin_33 = getitem_141.repeat_interleave(2, dim=-1)
        getitem_141 = None
        float_65 = query_states_30.float()
        mul_140 = float_65 * cos_33
        float_65 = None
        x1_30 = query_states_30[(Ellipsis, slice(0, None, 2))]
        x2_30 = query_states_30[(Ellipsis, slice(1, None, 2))]
        query_states_30 = None
        neg_30 = -x2_30
        x2_30 = None
        stack_30 = torch.stack((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        flatten_30 = stack_30.flatten(-2)
        stack_30 = None
        float_66 = flatten_30.float()
        flatten_30 = None
        mul_141 = float_66 * sin_33
        float_66 = None
        q_embed_15 = mul_140 + mul_141
        mul_140 = mul_141 = None
        float_67 = key_states_30.float()
        mul_142 = float_67 * cos_33
        float_67 = cos_33 = None
        x1_31 = key_states_30[(Ellipsis, slice(0, None, 2))]
        x2_31 = key_states_30[(Ellipsis, slice(1, None, 2))]
        key_states_30 = None
        neg_31 = -x2_31
        x2_31 = None
        stack_31 = torch.stack((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        flatten_31 = stack_31.flatten(-2)
        stack_31 = None
        float_68 = flatten_31.float()
        flatten_31 = None
        mul_143 = float_68 * sin_33
        float_68 = sin_33 = None
        k_embed_15 = mul_142 + mul_143
        mul_142 = mul_143 = None
        query_states_31 = q_embed_15.to(torch.bfloat16)
        q_embed_15 = None
        key_states_31 = k_embed_15.to(torch.bfloat16)
        k_embed_15 = None
        getitem_146 = key_states_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_153 = getitem_146.expand(1, 2, 8, 2, 128)
        getitem_146 = None
        key_30 = hidden_states_153.reshape(1, 16, 2, 128)
        hidden_states_153 = None
        getitem_147 = value_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_154 = getitem_147.expand(1, 2, 8, 2, 128)
        getitem_147 = None
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
        query_15 = query_states_31.contiguous()
        query_states_31 = None
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
        add_94 = variance_31 + 1e-05
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_157 = hidden_states_156 * rsqrt_31
        hidden_states_156 = rsqrt_31 = None
        to_97 = hidden_states_157.to(torch.bfloat16)
        hidden_states_157 = None
        hidden_states_158 = (
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_97
        )
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_97
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
        add_96 = variance_32 + 1e-05
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_161 = hidden_states_160 * rsqrt_32
        hidden_states_160 = rsqrt_32 = None
        to_99 = hidden_states_161.to(torch.bfloat16)
        hidden_states_161 = None
        hidden_states_162 = (
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
            * to_99
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            to_99
        ) = None
        linear_112 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_112.view((1, 2, -1, 128))
        linear_112 = None
        query_states_32 = view_49.transpose(1, 2)
        view_49 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_50 = linear_113.view((1, 2, -1, 128))
        linear_113 = None
        key_states_32 = view_50.transpose(1, 2)
        view_50 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_162 = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_51 = linear_114.view((1, 2, -1, 128))
        linear_114 = None
        value_states_16 = view_51.transpose(1, 2)
        view_51 = None
        cos_34 = cos_1.unsqueeze(1)
        sin_34 = sin_1.unsqueeze(1)
        getitem_149 = cos_34[(Ellipsis, slice(None, 64, None))]
        cos_34 = None
        cos_35 = getitem_149.repeat_interleave(2, dim=-1)
        getitem_149 = None
        getitem_150 = sin_34[(Ellipsis, slice(None, 64, None))]
        sin_34 = None
        sin_35 = getitem_150.repeat_interleave(2, dim=-1)
        getitem_150 = None
        float_69 = query_states_32.float()
        mul_149 = float_69 * cos_35
        float_69 = None
        x1_32 = query_states_32[(Ellipsis, slice(0, None, 2))]
        x2_32 = query_states_32[(Ellipsis, slice(1, None, 2))]
        query_states_32 = None
        neg_32 = -x2_32
        x2_32 = None
        stack_32 = torch.stack((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        flatten_32 = stack_32.flatten(-2)
        stack_32 = None
        float_70 = flatten_32.float()
        flatten_32 = None
        mul_150 = float_70 * sin_35
        float_70 = None
        q_embed_16 = mul_149 + mul_150
        mul_149 = mul_150 = None
        float_71 = key_states_32.float()
        mul_151 = float_71 * cos_35
        float_71 = cos_35 = None
        x1_33 = key_states_32[(Ellipsis, slice(0, None, 2))]
        x2_33 = key_states_32[(Ellipsis, slice(1, None, 2))]
        key_states_32 = None
        neg_33 = -x2_33
        x2_33 = None
        stack_33 = torch.stack((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        flatten_33 = stack_33.flatten(-2)
        stack_33 = None
        float_72 = flatten_33.float()
        flatten_33 = None
        mul_152 = float_72 * sin_35
        float_72 = sin_35 = None
        k_embed_16 = mul_151 + mul_152
        mul_151 = mul_152 = None
        query_states_33 = q_embed_16.to(torch.bfloat16)
        q_embed_16 = None
        key_states_33 = k_embed_16.to(torch.bfloat16)
        k_embed_16 = None
        getitem_155 = key_states_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_163 = getitem_155.expand(1, 2, 8, 2, 128)
        getitem_155 = None
        key_32 = hidden_states_163.reshape(1, 16, 2, 128)
        hidden_states_163 = None
        getitem_156 = value_states_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_164 = getitem_156.expand(1, 2, 8, 2, 128)
        getitem_156 = None
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
        query_16 = query_states_33.contiguous()
        query_states_33 = None
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
        add_100 = variance_33 + 1e-05
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_100)
        add_100 = None
        hidden_states_167 = hidden_states_166 * rsqrt_33
        hidden_states_166 = rsqrt_33 = None
        to_103 = hidden_states_167.to(torch.bfloat16)
        hidden_states_167 = None
        hidden_states_168 = (
            l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_103
        )
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_103
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
        add_102 = variance_34 + 1e-05
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_171 = hidden_states_170 * rsqrt_34
        hidden_states_170 = rsqrt_34 = None
        to_105 = hidden_states_171.to(torch.bfloat16)
        hidden_states_171 = None
        hidden_states_172 = (
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
            * to_105
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            to_105
        ) = None
        linear_119 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_119.view((1, 2, -1, 128))
        linear_119 = None
        query_states_34 = view_52.transpose(1, 2)
        view_52 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_53 = linear_120.view((1, 2, -1, 128))
        linear_120 = None
        key_states_34 = view_53.transpose(1, 2)
        view_53 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_172 = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_54 = linear_121.view((1, 2, -1, 128))
        linear_121 = None
        value_states_17 = view_54.transpose(1, 2)
        view_54 = None
        cos_36 = cos_1.unsqueeze(1)
        cos_1 = None
        sin_36 = sin_1.unsqueeze(1)
        sin_1 = None
        getitem_158 = cos_36[(Ellipsis, slice(None, 64, None))]
        cos_36 = None
        cos_37 = getitem_158.repeat_interleave(2, dim=-1)
        getitem_158 = None
        getitem_159 = sin_36[(Ellipsis, slice(None, 64, None))]
        sin_36 = None
        sin_37 = getitem_159.repeat_interleave(2, dim=-1)
        getitem_159 = None
        float_73 = query_states_34.float()
        mul_158 = float_73 * cos_37
        float_73 = None
        x1_34 = query_states_34[(Ellipsis, slice(0, None, 2))]
        x2_34 = query_states_34[(Ellipsis, slice(1, None, 2))]
        query_states_34 = None
        neg_34 = -x2_34
        x2_34 = None
        stack_34 = torch.stack((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        flatten_34 = stack_34.flatten(-2)
        stack_34 = None
        float_74 = flatten_34.float()
        flatten_34 = None
        mul_159 = float_74 * sin_37
        float_74 = None
        q_embed_17 = mul_158 + mul_159
        mul_158 = mul_159 = None
        float_75 = key_states_34.float()
        mul_160 = float_75 * cos_37
        float_75 = cos_37 = None
        x1_35 = key_states_34[(Ellipsis, slice(0, None, 2))]
        x2_35 = key_states_34[(Ellipsis, slice(1, None, 2))]
        key_states_34 = None
        neg_35 = -x2_35
        x2_35 = None
        stack_35 = torch.stack((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        flatten_35 = stack_35.flatten(-2)
        stack_35 = None
        float_76 = flatten_35.float()
        flatten_35 = None
        mul_161 = float_76 * sin_37
        float_76 = sin_37 = None
        k_embed_17 = mul_160 + mul_161
        mul_160 = mul_161 = None
        query_states_35 = q_embed_17.to(torch.bfloat16)
        q_embed_17 = None
        key_states_35 = k_embed_17.to(torch.bfloat16)
        k_embed_17 = None
        getitem_164 = key_states_35[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_173 = getitem_164.expand(1, 2, 8, 2, 128)
        getitem_164 = None
        key_34 = hidden_states_173.reshape(1, 16, 2, 128)
        hidden_states_173 = None
        getitem_165 = value_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_174 = getitem_165.expand(1, 2, 8, 2, 128)
        getitem_165 = None
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
        causal_mask_2 = None
        query_17 = query_states_35.contiguous()
        query_states_35 = None
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
        add_106 = variance_35 + 1e-05
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_177 = hidden_states_176 * rsqrt_35
        hidden_states_176 = rsqrt_35 = None
        to_109 = hidden_states_177.to(torch.bfloat16)
        hidden_states_177 = None
        hidden_states_178 = (
            l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_109
        )
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_109
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
        hidden_states_179 = None
        pow_37 = hidden_states_180.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_108 = variance_36 + 1e-05
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_181 = hidden_states_180 * rsqrt_36
        hidden_states_180 = rsqrt_36 = None
        to_111 = hidden_states_181.to(torch.bfloat16)
        hidden_states_181 = None
        hidden_states_182 = l_self_modules_norm_parameters_weight_ * to_111
        l_self_modules_norm_parameters_weight_ = to_111 = None
        return (
            value_states,
            key_states_1,
            value_states_1,
            key_states_3,
            value_states_2,
            key_states_5,
            value_states_3,
            key_states_7,
            value_states_4,
            key_states_9,
            value_states_5,
            key_states_11,
            value_states_6,
            key_states_13,
            value_states_7,
            key_states_15,
            value_states_8,
            key_states_17,
            value_states_9,
            key_states_19,
            value_states_10,
            key_states_21,
            value_states_11,
            key_states_23,
            value_states_12,
            key_states_25,
            value_states_13,
            key_states_27,
            value_states_14,
            key_states_29,
            value_states_15,
            key_states_31,
            value_states_16,
            key_states_33,
            value_states_17,
            key_states_35,
            hidden_states_182,
        )
