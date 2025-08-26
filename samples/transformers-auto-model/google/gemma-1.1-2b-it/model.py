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
        cache_position = torch.arange(0, 3, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_attention_mask_ = None
        mask_indices = torch.arange(3, device=device(type="cuda", index=0))
        mask_indices += 0
        mask_indices_1 = mask_indices
        mask_indices = None
        local_padding_mask = attention_mask[(slice(None, None, None), mask_indices_1)]
        attention_mask = mask_indices_1 = None
        kv_arange = torch.arange(3, device=device(type="cuda", index=0))
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
        normalizer = torch.tensor(45.254833995939045, dtype=torch.bfloat16)
        hidden_states = l_inputs_embeds_ * normalizer
        l_inputs_embeds_ = normalizer = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        float_5 = hidden_states.float()
        pow_1 = float_5.pow(2)
        mean = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = mean + 1e-06
        mean = None
        rsqrt = torch.rsqrt(add)
        add = None
        output = float_5 * rsqrt
        float_5 = rsqrt = None
        float_6 = (
            l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_1 = 1.0 + float_6
        float_6 = None
        output_1 = output * add_1
        output = add_1 = None
        hidden_states_1 = output_1.type_as(hidden_states)
        output_1 = None
        linear = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_1 = linear.view((1, 3, -1, 256))
        linear = None
        query_states = view_1.transpose(1, 2)
        view_1 = None
        linear_1 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_2 = linear_1.view((1, 3, -1, 256))
        linear_1 = None
        key_states = view_2.transpose(1, 2)
        view_2 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_1,
            l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_1 = l_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_3 = linear_2.view((1, 3, -1, 256))
        linear_2 = None
        value_states = view_3.transpose(1, 2)
        view_3 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_6 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 128, None))]
        x2 = query_states[(Ellipsis, slice(128, None, None))]
        query_states = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_7 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_6 + mul_7
        mul_6 = mul_7 = None
        mul_8 = key_states * cos_3
        cos_3 = None
        x1_1 = key_states[(Ellipsis, slice(None, 128, None))]
        x2_1 = key_states[(Ellipsis, slice(128, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_9 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_8 + mul_9
        mul_8 = mul_9 = None
        getitem_9 = k_embed[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_2 = getitem_9.expand(1, 1, 8, 3, 256)
        getitem_9 = None
        key = hidden_states_2.reshape(1, 8, 3, 256)
        hidden_states_2 = None
        getitem_10 = value_states[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_3 = getitem_10.expand(1, 1, 8, 3, 256)
        getitem_10 = None
        value = hidden_states_3.reshape(1, 8, 3, 256)
        hidden_states_3 = None
        attention_mask_1 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query = key_1 = value_1 = attention_mask_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape_2 = attn_output_1.reshape(1, 3, -1)
        attn_output_1 = None
        attn_output_2 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_4 = hidden_states + attn_output_3
        hidden_states = attn_output_3 = None
        float_7 = hidden_states_4.float()
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
            l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_6 = 1.0 + float_8
        float_8 = None
        output_3 = output_2 * add_6
        output_2 = add_6 = None
        hidden_states_5 = output_3.type_as(hidden_states_4)
        output_3 = None
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
        mul_12 = gelu * linear_5
        gelu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_12,
            l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_12 = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_6 = hidden_states_4 + down_proj
        hidden_states_4 = down_proj = None
        float_9 = hidden_states_6.float()
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
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_9 = 1.0 + float_10
        float_10 = None
        output_5 = output_4 * add_9
        output_4 = add_9 = None
        hidden_states_7 = output_5.type_as(hidden_states_6)
        output_5 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_7.view((1, 3, -1, 256))
        linear_7 = None
        query_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_5 = linear_8.view((1, 3, -1, 256))
        linear_8 = None
        key_states_1 = view_5.transpose(1, 2)
        view_5 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_7 = l_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_6 = linear_9.view((1, 3, -1, 256))
        linear_9 = None
        value_states_1 = view_6.transpose(1, 2)
        view_6 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_15 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 128, None))]
        x2_2 = query_states_1[(Ellipsis, slice(128, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_16 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_15 + mul_16
        mul_15 = mul_16 = None
        mul_17 = key_states_1 * cos_4
        cos_4 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 128, None))]
        x2_3 = key_states_1[(Ellipsis, slice(128, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_18 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_17 + mul_18
        mul_17 = mul_18 = None
        getitem_16 = k_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_8 = getitem_16.expand(1, 1, 8, 3, 256)
        getitem_16 = None
        key_2 = hidden_states_8.reshape(1, 8, 3, 256)
        hidden_states_8 = None
        getitem_17 = value_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_9 = getitem_17.expand(1, 1, 8, 3, 256)
        getitem_17 = None
        value_2 = hidden_states_9.reshape(1, 8, 3, 256)
        hidden_states_9 = None
        attention_mask_2 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_1 = key_3 = value_3 = attention_mask_2 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_5 = attn_output_5.reshape(1, 3, -1)
        attn_output_5 = None
        attn_output_6 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_10 = hidden_states_6 + attn_output_7
        hidden_states_6 = attn_output_7 = None
        float_11 = hidden_states_10.float()
        pow_4 = float_11.pow(2)
        mean_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_13 = mean_3 + 1e-06
        mean_3 = None
        rsqrt_3 = torch.rsqrt(add_13)
        add_13 = None
        output_6 = float_11 * rsqrt_3
        float_11 = rsqrt_3 = None
        float_12 = (
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_14 = 1.0 + float_12
        float_12 = None
        output_7 = output_6 * add_14
        output_6 = add_14 = None
        hidden_states_11 = output_7.type_as(hidden_states_10)
        output_7 = None
        linear_11 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_1 = torch._C._nn.gelu(linear_11, approximate="tanh")
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_11 = l_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_21 = gelu_1 * linear_12
        gelu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_21,
            l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_21 = l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_12 = hidden_states_10 + down_proj_1
        hidden_states_10 = down_proj_1 = None
        float_13 = hidden_states_12.float()
        pow_5 = float_13.pow(2)
        mean_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_16 = mean_4 + 1e-06
        mean_4 = None
        rsqrt_4 = torch.rsqrt(add_16)
        add_16 = None
        output_8 = float_13 * rsqrt_4
        float_13 = rsqrt_4 = None
        float_14 = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_17 = 1.0 + float_14
        float_14 = None
        output_9 = output_8 * add_17
        output_8 = add_17 = None
        hidden_states_13 = output_9.type_as(hidden_states_12)
        output_9 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_14.view((1, 3, -1, 256))
        linear_14 = None
        query_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_8 = linear_15.view((1, 3, -1, 256))
        linear_15 = None
        key_states_2 = view_8.transpose(1, 2)
        view_8 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_13 = l_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_9 = linear_16.view((1, 3, -1, 256))
        linear_16 = None
        value_states_2 = view_9.transpose(1, 2)
        view_9 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_24 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 128, None))]
        x2_4 = query_states_2[(Ellipsis, slice(128, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_25 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_24 + mul_25
        mul_24 = mul_25 = None
        mul_26 = key_states_2 * cos_5
        cos_5 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 128, None))]
        x2_5 = key_states_2[(Ellipsis, slice(128, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_27 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_26 + mul_27
        mul_26 = mul_27 = None
        getitem_23 = k_embed_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_14 = getitem_23.expand(1, 1, 8, 3, 256)
        getitem_23 = None
        key_4 = hidden_states_14.reshape(1, 8, 3, 256)
        hidden_states_14 = None
        getitem_24 = value_states_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_15 = getitem_24.expand(1, 1, 8, 3, 256)
        getitem_24 = None
        value_4 = hidden_states_15.reshape(1, 8, 3, 256)
        hidden_states_15 = None
        attention_mask_3 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_2 = key_5 = value_5 = attention_mask_3 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_8 = attn_output_9.reshape(1, 3, -1)
        attn_output_9 = None
        attn_output_10 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_16 = hidden_states_12 + attn_output_11
        hidden_states_12 = attn_output_11 = None
        float_15 = hidden_states_16.float()
        pow_6 = float_15.pow(2)
        mean_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_21 = mean_5 + 1e-06
        mean_5 = None
        rsqrt_5 = torch.rsqrt(add_21)
        add_21 = None
        output_10 = float_15 * rsqrt_5
        float_15 = rsqrt_5 = None
        float_16 = (
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_22 = 1.0 + float_16
        float_16 = None
        output_11 = output_10 * add_22
        output_10 = add_22 = None
        hidden_states_17 = output_11.type_as(hidden_states_16)
        output_11 = None
        linear_18 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_2 = torch._C._nn.gelu(linear_18, approximate="tanh")
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_17 = l_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_30 = gelu_2 * linear_19
        gelu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_30,
            l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_30 = l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_18 = hidden_states_16 + down_proj_2
        hidden_states_16 = down_proj_2 = None
        float_17 = hidden_states_18.float()
        pow_7 = float_17.pow(2)
        mean_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_24 = mean_6 + 1e-06
        mean_6 = None
        rsqrt_6 = torch.rsqrt(add_24)
        add_24 = None
        output_12 = float_17 * rsqrt_6
        float_17 = rsqrt_6 = None
        float_18 = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_25 = 1.0 + float_18
        float_18 = None
        output_13 = output_12 * add_25
        output_12 = add_25 = None
        hidden_states_19 = output_13.type_as(hidden_states_18)
        output_13 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_21.view((1, 3, -1, 256))
        linear_21 = None
        query_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_11 = linear_22.view((1, 3, -1, 256))
        linear_22 = None
        key_states_3 = view_11.transpose(1, 2)
        view_11 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_19,
            l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_19 = l_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_12 = linear_23.view((1, 3, -1, 256))
        linear_23 = None
        value_states_3 = view_12.transpose(1, 2)
        view_12 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_33 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 128, None))]
        x2_6 = query_states_3[(Ellipsis, slice(128, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_34 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_33 + mul_34
        mul_33 = mul_34 = None
        mul_35 = key_states_3 * cos_6
        cos_6 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 128, None))]
        x2_7 = key_states_3[(Ellipsis, slice(128, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_36 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_35 + mul_36
        mul_35 = mul_36 = None
        getitem_30 = k_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_20 = getitem_30.expand(1, 1, 8, 3, 256)
        getitem_30 = None
        key_6 = hidden_states_20.reshape(1, 8, 3, 256)
        hidden_states_20 = None
        getitem_31 = value_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_21 = getitem_31.expand(1, 1, 8, 3, 256)
        getitem_31 = None
        value_6 = hidden_states_21.reshape(1, 8, 3, 256)
        hidden_states_21 = None
        attention_mask_4 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_3 = key_7 = value_7 = attention_mask_4 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_11 = attn_output_13.reshape(1, 3, -1)
        attn_output_13 = None
        attn_output_14 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_22 = hidden_states_18 + attn_output_15
        hidden_states_18 = attn_output_15 = None
        float_19 = hidden_states_22.float()
        pow_8 = float_19.pow(2)
        mean_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_29 = mean_7 + 1e-06
        mean_7 = None
        rsqrt_7 = torch.rsqrt(add_29)
        add_29 = None
        output_14 = float_19 * rsqrt_7
        float_19 = rsqrt_7 = None
        float_20 = (
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_30 = 1.0 + float_20
        float_20 = None
        output_15 = output_14 * add_30
        output_14 = add_30 = None
        hidden_states_23 = output_15.type_as(hidden_states_22)
        output_15 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_3 = torch._C._nn.gelu(linear_25, approximate="tanh")
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_23 = l_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_39 = gelu_3 * linear_26
        gelu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_39,
            l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_39 = l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_24 = hidden_states_22 + down_proj_3
        hidden_states_22 = down_proj_3 = None
        float_21 = hidden_states_24.float()
        pow_9 = float_21.pow(2)
        mean_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_32 = mean_8 + 1e-06
        mean_8 = None
        rsqrt_8 = torch.rsqrt(add_32)
        add_32 = None
        output_16 = float_21 * rsqrt_8
        float_21 = rsqrt_8 = None
        float_22 = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_33 = 1.0 + float_22
        float_22 = None
        output_17 = output_16 * add_33
        output_16 = add_33 = None
        hidden_states_25 = output_17.type_as(hidden_states_24)
        output_17 = None
        linear_28 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_28.view((1, 3, -1, 256))
        linear_28 = None
        query_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_14 = linear_29.view((1, 3, -1, 256))
        linear_29 = None
        key_states_4 = view_14.transpose(1, 2)
        view_14 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_25 = l_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_15 = linear_30.view((1, 3, -1, 256))
        linear_30 = None
        value_states_4 = view_15.transpose(1, 2)
        view_15 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_42 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 128, None))]
        x2_8 = query_states_4[(Ellipsis, slice(128, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_43 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_42 + mul_43
        mul_42 = mul_43 = None
        mul_44 = key_states_4 * cos_7
        cos_7 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 128, None))]
        x2_9 = key_states_4[(Ellipsis, slice(128, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_45 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_44 + mul_45
        mul_44 = mul_45 = None
        getitem_37 = k_embed_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_26 = getitem_37.expand(1, 1, 8, 3, 256)
        getitem_37 = None
        key_8 = hidden_states_26.reshape(1, 8, 3, 256)
        hidden_states_26 = None
        getitem_38 = value_states_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_27 = getitem_38.expand(1, 1, 8, 3, 256)
        getitem_38 = None
        value_8 = hidden_states_27.reshape(1, 8, 3, 256)
        hidden_states_27 = None
        attention_mask_5 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_4 = key_9 = value_9 = attention_mask_5 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_14 = attn_output_17.reshape(1, 3, -1)
        attn_output_17 = None
        attn_output_18 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_28 = hidden_states_24 + attn_output_19
        hidden_states_24 = attn_output_19 = None
        float_23 = hidden_states_28.float()
        pow_10 = float_23.pow(2)
        mean_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_37 = mean_9 + 1e-06
        mean_9 = None
        rsqrt_9 = torch.rsqrt(add_37)
        add_37 = None
        output_18 = float_23 * rsqrt_9
        float_23 = rsqrt_9 = None
        float_24 = (
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_38 = 1.0 + float_24
        float_24 = None
        output_19 = output_18 * add_38
        output_18 = add_38 = None
        hidden_states_29 = output_19.type_as(hidden_states_28)
        output_19 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_4 = torch._C._nn.gelu(linear_32, approximate="tanh")
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_29 = l_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_48 = gelu_4 * linear_33
        gelu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_48,
            l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_48 = l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_30 = hidden_states_28 + down_proj_4
        hidden_states_28 = down_proj_4 = None
        float_25 = hidden_states_30.float()
        pow_11 = float_25.pow(2)
        mean_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_40 = mean_10 + 1e-06
        mean_10 = None
        rsqrt_10 = torch.rsqrt(add_40)
        add_40 = None
        output_20 = float_25 * rsqrt_10
        float_25 = rsqrt_10 = None
        float_26 = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_41 = 1.0 + float_26
        float_26 = None
        output_21 = output_20 * add_41
        output_20 = add_41 = None
        hidden_states_31 = output_21.type_as(hidden_states_30)
        output_21 = None
        linear_35 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_35.view((1, 3, -1, 256))
        linear_35 = None
        query_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_17 = linear_36.view((1, 3, -1, 256))
        linear_36 = None
        key_states_5 = view_17.transpose(1, 2)
        view_17 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_31,
            l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_31 = l_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_18 = linear_37.view((1, 3, -1, 256))
        linear_37 = None
        value_states_5 = view_18.transpose(1, 2)
        view_18 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_51 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 128, None))]
        x2_10 = query_states_5[(Ellipsis, slice(128, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_52 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_51 + mul_52
        mul_51 = mul_52 = None
        mul_53 = key_states_5 * cos_8
        cos_8 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 128, None))]
        x2_11 = key_states_5[(Ellipsis, slice(128, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_54 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_53 + mul_54
        mul_53 = mul_54 = None
        getitem_44 = k_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_32 = getitem_44.expand(1, 1, 8, 3, 256)
        getitem_44 = None
        key_10 = hidden_states_32.reshape(1, 8, 3, 256)
        hidden_states_32 = None
        getitem_45 = value_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_33 = getitem_45.expand(1, 1, 8, 3, 256)
        getitem_45 = None
        value_10 = hidden_states_33.reshape(1, 8, 3, 256)
        hidden_states_33 = None
        attention_mask_6 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_5 = key_11 = value_11 = attention_mask_6 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_17 = attn_output_21.reshape(1, 3, -1)
        attn_output_21 = None
        attn_output_22 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_34 = hidden_states_30 + attn_output_23
        hidden_states_30 = attn_output_23 = None
        float_27 = hidden_states_34.float()
        pow_12 = float_27.pow(2)
        mean_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_45 = mean_11 + 1e-06
        mean_11 = None
        rsqrt_11 = torch.rsqrt(add_45)
        add_45 = None
        output_22 = float_27 * rsqrt_11
        float_27 = rsqrt_11 = None
        float_28 = (
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_46 = 1.0 + float_28
        float_28 = None
        output_23 = output_22 * add_46
        output_22 = add_46 = None
        hidden_states_35 = output_23.type_as(hidden_states_34)
        output_23 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_5 = torch._C._nn.gelu(linear_39, approximate="tanh")
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_35 = l_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_57 = gelu_5 * linear_40
        gelu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_57,
            l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_57 = l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_36 = hidden_states_34 + down_proj_5
        hidden_states_34 = down_proj_5 = None
        float_29 = hidden_states_36.float()
        pow_13 = float_29.pow(2)
        mean_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_48 = mean_12 + 1e-06
        mean_12 = None
        rsqrt_12 = torch.rsqrt(add_48)
        add_48 = None
        output_24 = float_29 * rsqrt_12
        float_29 = rsqrt_12 = None
        float_30 = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_49 = 1.0 + float_30
        float_30 = None
        output_25 = output_24 * add_49
        output_24 = add_49 = None
        hidden_states_37 = output_25.type_as(hidden_states_36)
        output_25 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_42.view((1, 3, -1, 256))
        linear_42 = None
        query_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_20 = linear_43.view((1, 3, -1, 256))
        linear_43 = None
        key_states_6 = view_20.transpose(1, 2)
        view_20 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_37 = l_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_21 = linear_44.view((1, 3, -1, 256))
        linear_44 = None
        value_states_6 = view_21.transpose(1, 2)
        view_21 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_60 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 128, None))]
        x2_12 = query_states_6[(Ellipsis, slice(128, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_61 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_60 + mul_61
        mul_60 = mul_61 = None
        mul_62 = key_states_6 * cos_9
        cos_9 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 128, None))]
        x2_13 = key_states_6[(Ellipsis, slice(128, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_63 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_62 + mul_63
        mul_62 = mul_63 = None
        getitem_51 = k_embed_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_38 = getitem_51.expand(1, 1, 8, 3, 256)
        getitem_51 = None
        key_12 = hidden_states_38.reshape(1, 8, 3, 256)
        hidden_states_38 = None
        getitem_52 = value_states_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_39 = getitem_52.expand(1, 1, 8, 3, 256)
        getitem_52 = None
        value_12 = hidden_states_39.reshape(1, 8, 3, 256)
        hidden_states_39 = None
        attention_mask_7 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_6 = key_13 = value_13 = attention_mask_7 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_20 = attn_output_25.reshape(1, 3, -1)
        attn_output_25 = None
        attn_output_26 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_40 = hidden_states_36 + attn_output_27
        hidden_states_36 = attn_output_27 = None
        float_31 = hidden_states_40.float()
        pow_14 = float_31.pow(2)
        mean_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_53 = mean_13 + 1e-06
        mean_13 = None
        rsqrt_13 = torch.rsqrt(add_53)
        add_53 = None
        output_26 = float_31 * rsqrt_13
        float_31 = rsqrt_13 = None
        float_32 = (
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_54 = 1.0 + float_32
        float_32 = None
        output_27 = output_26 * add_54
        output_26 = add_54 = None
        hidden_states_41 = output_27.type_as(hidden_states_40)
        output_27 = None
        linear_46 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_6 = torch._C._nn.gelu(linear_46, approximate="tanh")
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_41 = l_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_66 = gelu_6 * linear_47
        gelu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_66,
            l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_66 = l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_42 = hidden_states_40 + down_proj_6
        hidden_states_40 = down_proj_6 = None
        float_33 = hidden_states_42.float()
        pow_15 = float_33.pow(2)
        mean_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_56 = mean_14 + 1e-06
        mean_14 = None
        rsqrt_14 = torch.rsqrt(add_56)
        add_56 = None
        output_28 = float_33 * rsqrt_14
        float_33 = rsqrt_14 = None
        float_34 = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_57 = 1.0 + float_34
        float_34 = None
        output_29 = output_28 * add_57
        output_28 = add_57 = None
        hidden_states_43 = output_29.type_as(hidden_states_42)
        output_29 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_49.view((1, 3, -1, 256))
        linear_49 = None
        query_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_23 = linear_50.view((1, 3, -1, 256))
        linear_50 = None
        key_states_7 = view_23.transpose(1, 2)
        view_23 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_43,
            l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_43 = l_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_24 = linear_51.view((1, 3, -1, 256))
        linear_51 = None
        value_states_7 = view_24.transpose(1, 2)
        view_24 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_69 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 128, None))]
        x2_14 = query_states_7[(Ellipsis, slice(128, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_70 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_69 + mul_70
        mul_69 = mul_70 = None
        mul_71 = key_states_7 * cos_10
        cos_10 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 128, None))]
        x2_15 = key_states_7[(Ellipsis, slice(128, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_72 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_71 + mul_72
        mul_71 = mul_72 = None
        getitem_58 = k_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_44 = getitem_58.expand(1, 1, 8, 3, 256)
        getitem_58 = None
        key_14 = hidden_states_44.reshape(1, 8, 3, 256)
        hidden_states_44 = None
        getitem_59 = value_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_45 = getitem_59.expand(1, 1, 8, 3, 256)
        getitem_59 = None
        value_14 = hidden_states_45.reshape(1, 8, 3, 256)
        hidden_states_45 = None
        attention_mask_8 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_7 = key_15 = value_15 = attention_mask_8 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_23 = attn_output_29.reshape(1, 3, -1)
        attn_output_29 = None
        attn_output_30 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_46 = hidden_states_42 + attn_output_31
        hidden_states_42 = attn_output_31 = None
        float_35 = hidden_states_46.float()
        pow_16 = float_35.pow(2)
        mean_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_61 = mean_15 + 1e-06
        mean_15 = None
        rsqrt_15 = torch.rsqrt(add_61)
        add_61 = None
        output_30 = float_35 * rsqrt_15
        float_35 = rsqrt_15 = None
        float_36 = (
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_62 = 1.0 + float_36
        float_36 = None
        output_31 = output_30 * add_62
        output_30 = add_62 = None
        hidden_states_47 = output_31.type_as(hidden_states_46)
        output_31 = None
        linear_53 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_7 = torch._C._nn.gelu(linear_53, approximate="tanh")
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_75 = gelu_7 * linear_54
        gelu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_75,
            l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_75 = l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_48 = hidden_states_46 + down_proj_7
        hidden_states_46 = down_proj_7 = None
        float_37 = hidden_states_48.float()
        pow_17 = float_37.pow(2)
        mean_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_64 = mean_16 + 1e-06
        mean_16 = None
        rsqrt_16 = torch.rsqrt(add_64)
        add_64 = None
        output_32 = float_37 * rsqrt_16
        float_37 = rsqrt_16 = None
        float_38 = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_65 = 1.0 + float_38
        float_38 = None
        output_33 = output_32 * add_65
        output_32 = add_65 = None
        hidden_states_49 = output_33.type_as(hidden_states_48)
        output_33 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_56.view((1, 3, -1, 256))
        linear_56 = None
        query_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_26 = linear_57.view((1, 3, -1, 256))
        linear_57 = None
        key_states_8 = view_26.transpose(1, 2)
        view_26 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_49,
            l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_49 = l_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_27 = linear_58.view((1, 3, -1, 256))
        linear_58 = None
        value_states_8 = view_27.transpose(1, 2)
        view_27 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_78 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 128, None))]
        x2_16 = query_states_8[(Ellipsis, slice(128, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_79 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_78 + mul_79
        mul_78 = mul_79 = None
        mul_80 = key_states_8 * cos_11
        cos_11 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 128, None))]
        x2_17 = key_states_8[(Ellipsis, slice(128, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_81 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_80 + mul_81
        mul_80 = mul_81 = None
        getitem_65 = k_embed_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_50 = getitem_65.expand(1, 1, 8, 3, 256)
        getitem_65 = None
        key_16 = hidden_states_50.reshape(1, 8, 3, 256)
        hidden_states_50 = None
        getitem_66 = value_states_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_51 = getitem_66.expand(1, 1, 8, 3, 256)
        getitem_66 = None
        value_16 = hidden_states_51.reshape(1, 8, 3, 256)
        hidden_states_51 = None
        attention_mask_9 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_8 = key_17 = value_17 = attention_mask_9 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_26 = attn_output_33.reshape(1, 3, -1)
        attn_output_33 = None
        attn_output_34 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_52 = hidden_states_48 + attn_output_35
        hidden_states_48 = attn_output_35 = None
        float_39 = hidden_states_52.float()
        pow_18 = float_39.pow(2)
        mean_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_69 = mean_17 + 1e-06
        mean_17 = None
        rsqrt_17 = torch.rsqrt(add_69)
        add_69 = None
        output_34 = float_39 * rsqrt_17
        float_39 = rsqrt_17 = None
        float_40 = (
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_70 = 1.0 + float_40
        float_40 = None
        output_35 = output_34 * add_70
        output_34 = add_70 = None
        hidden_states_53 = output_35.type_as(hidden_states_52)
        output_35 = None
        linear_60 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_8 = torch._C._nn.gelu(linear_60, approximate="tanh")
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_53 = l_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_84 = gelu_8 * linear_61
        gelu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_84,
            l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_84 = l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_54 = hidden_states_52 + down_proj_8
        hidden_states_52 = down_proj_8 = None
        float_41 = hidden_states_54.float()
        pow_19 = float_41.pow(2)
        mean_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_72 = mean_18 + 1e-06
        mean_18 = None
        rsqrt_18 = torch.rsqrt(add_72)
        add_72 = None
        output_36 = float_41 * rsqrt_18
        float_41 = rsqrt_18 = None
        float_42 = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_73 = 1.0 + float_42
        float_42 = None
        output_37 = output_36 * add_73
        output_36 = add_73 = None
        hidden_states_55 = output_37.type_as(hidden_states_54)
        output_37 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_63.view((1, 3, -1, 256))
        linear_63 = None
        query_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_29 = linear_64.view((1, 3, -1, 256))
        linear_64 = None
        key_states_9 = view_29.transpose(1, 2)
        view_29 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_55 = l_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_30 = linear_65.view((1, 3, -1, 256))
        linear_65 = None
        value_states_9 = view_30.transpose(1, 2)
        view_30 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_87 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 128, None))]
        x2_18 = query_states_9[(Ellipsis, slice(128, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_88 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_87 + mul_88
        mul_87 = mul_88 = None
        mul_89 = key_states_9 * cos_12
        cos_12 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 128, None))]
        x2_19 = key_states_9[(Ellipsis, slice(128, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_90 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_89 + mul_90
        mul_89 = mul_90 = None
        getitem_72 = k_embed_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_56 = getitem_72.expand(1, 1, 8, 3, 256)
        getitem_72 = None
        key_18 = hidden_states_56.reshape(1, 8, 3, 256)
        hidden_states_56 = None
        getitem_73 = value_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_57 = getitem_73.expand(1, 1, 8, 3, 256)
        getitem_73 = None
        value_18 = hidden_states_57.reshape(1, 8, 3, 256)
        hidden_states_57 = None
        attention_mask_10 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_9 = key_19 = value_19 = attention_mask_10 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_29 = attn_output_37.reshape(1, 3, -1)
        attn_output_37 = None
        attn_output_38 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_58 = hidden_states_54 + attn_output_39
        hidden_states_54 = attn_output_39 = None
        float_43 = hidden_states_58.float()
        pow_20 = float_43.pow(2)
        mean_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_77 = mean_19 + 1e-06
        mean_19 = None
        rsqrt_19 = torch.rsqrt(add_77)
        add_77 = None
        output_38 = float_43 * rsqrt_19
        float_43 = rsqrt_19 = None
        float_44 = (
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_78 = 1.0 + float_44
        float_44 = None
        output_39 = output_38 * add_78
        output_38 = add_78 = None
        hidden_states_59 = output_39.type_as(hidden_states_58)
        output_39 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_9 = torch._C._nn.gelu(linear_67, approximate="tanh")
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_59,
            l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_59 = l_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_93 = gelu_9 * linear_68
        gelu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_93,
            l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_93 = l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_60 = hidden_states_58 + down_proj_9
        hidden_states_58 = down_proj_9 = None
        float_45 = hidden_states_60.float()
        pow_21 = float_45.pow(2)
        mean_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_80 = mean_20 + 1e-06
        mean_20 = None
        rsqrt_20 = torch.rsqrt(add_80)
        add_80 = None
        output_40 = float_45 * rsqrt_20
        float_45 = rsqrt_20 = None
        float_46 = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_81 = 1.0 + float_46
        float_46 = None
        output_41 = output_40 * add_81
        output_40 = add_81 = None
        hidden_states_61 = output_41.type_as(hidden_states_60)
        output_41 = None
        linear_70 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_70.view((1, 3, -1, 256))
        linear_70 = None
        query_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_32 = linear_71.view((1, 3, -1, 256))
        linear_71 = None
        key_states_10 = view_32.transpose(1, 2)
        view_32 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_61 = l_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_33 = linear_72.view((1, 3, -1, 256))
        linear_72 = None
        value_states_10 = view_33.transpose(1, 2)
        view_33 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_96 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 128, None))]
        x2_20 = query_states_10[(Ellipsis, slice(128, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_97 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_96 + mul_97
        mul_96 = mul_97 = None
        mul_98 = key_states_10 * cos_13
        cos_13 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 128, None))]
        x2_21 = key_states_10[(Ellipsis, slice(128, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_99 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_98 + mul_99
        mul_98 = mul_99 = None
        getitem_79 = k_embed_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_62 = getitem_79.expand(1, 1, 8, 3, 256)
        getitem_79 = None
        key_20 = hidden_states_62.reshape(1, 8, 3, 256)
        hidden_states_62 = None
        getitem_80 = value_states_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_63 = getitem_80.expand(1, 1, 8, 3, 256)
        getitem_80 = None
        value_20 = hidden_states_63.reshape(1, 8, 3, 256)
        hidden_states_63 = None
        attention_mask_11 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_10 = key_21 = value_21 = attention_mask_11 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_32 = attn_output_41.reshape(1, 3, -1)
        attn_output_41 = None
        attn_output_42 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_64 = hidden_states_60 + attn_output_43
        hidden_states_60 = attn_output_43 = None
        float_47 = hidden_states_64.float()
        pow_22 = float_47.pow(2)
        mean_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_85 = mean_21 + 1e-06
        mean_21 = None
        rsqrt_21 = torch.rsqrt(add_85)
        add_85 = None
        output_42 = float_47 * rsqrt_21
        float_47 = rsqrt_21 = None
        float_48 = (
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_86 = 1.0 + float_48
        float_48 = None
        output_43 = output_42 * add_86
        output_42 = add_86 = None
        hidden_states_65 = output_43.type_as(hidden_states_64)
        output_43 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_10 = torch._C._nn.gelu(linear_74, approximate="tanh")
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_65 = l_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_102 = gelu_10 * linear_75
        gelu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_102,
            l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_102 = l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_66 = hidden_states_64 + down_proj_10
        hidden_states_64 = down_proj_10 = None
        float_49 = hidden_states_66.float()
        pow_23 = float_49.pow(2)
        mean_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_88 = mean_22 + 1e-06
        mean_22 = None
        rsqrt_22 = torch.rsqrt(add_88)
        add_88 = None
        output_44 = float_49 * rsqrt_22
        float_49 = rsqrt_22 = None
        float_50 = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_89 = 1.0 + float_50
        float_50 = None
        output_45 = output_44 * add_89
        output_44 = add_89 = None
        hidden_states_67 = output_45.type_as(hidden_states_66)
        output_45 = None
        linear_77 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_77.view((1, 3, -1, 256))
        linear_77 = None
        query_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_35 = linear_78.view((1, 3, -1, 256))
        linear_78 = None
        key_states_11 = view_35.transpose(1, 2)
        view_35 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_67 = l_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_36 = linear_79.view((1, 3, -1, 256))
        linear_79 = None
        value_states_11 = view_36.transpose(1, 2)
        view_36 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_105 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 128, None))]
        x2_22 = query_states_11[(Ellipsis, slice(128, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_106 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_105 + mul_106
        mul_105 = mul_106 = None
        mul_107 = key_states_11 * cos_14
        cos_14 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 128, None))]
        x2_23 = key_states_11[(Ellipsis, slice(128, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_108 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_107 + mul_108
        mul_107 = mul_108 = None
        getitem_86 = k_embed_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_68 = getitem_86.expand(1, 1, 8, 3, 256)
        getitem_86 = None
        key_22 = hidden_states_68.reshape(1, 8, 3, 256)
        hidden_states_68 = None
        getitem_87 = value_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_69 = getitem_87.expand(1, 1, 8, 3, 256)
        getitem_87 = None
        value_22 = hidden_states_69.reshape(1, 8, 3, 256)
        hidden_states_69 = None
        attention_mask_12 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_11 = key_23 = value_23 = attention_mask_12 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_35 = attn_output_45.reshape(1, 3, -1)
        attn_output_45 = None
        attn_output_46 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_70 = hidden_states_66 + attn_output_47
        hidden_states_66 = attn_output_47 = None
        float_51 = hidden_states_70.float()
        pow_24 = float_51.pow(2)
        mean_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_93 = mean_23 + 1e-06
        mean_23 = None
        rsqrt_23 = torch.rsqrt(add_93)
        add_93 = None
        output_46 = float_51 * rsqrt_23
        float_51 = rsqrt_23 = None
        float_52 = (
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_94 = 1.0 + float_52
        float_52 = None
        output_47 = output_46 * add_94
        output_46 = add_94 = None
        hidden_states_71 = output_47.type_as(hidden_states_70)
        output_47 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_11 = torch._C._nn.gelu(linear_81, approximate="tanh")
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_71,
            l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_71 = l_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_111 = gelu_11 * linear_82
        gelu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_111,
            l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_111 = l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_72 = hidden_states_70 + down_proj_11
        hidden_states_70 = down_proj_11 = None
        float_53 = hidden_states_72.float()
        pow_25 = float_53.pow(2)
        mean_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_96 = mean_24 + 1e-06
        mean_24 = None
        rsqrt_24 = torch.rsqrt(add_96)
        add_96 = None
        output_48 = float_53 * rsqrt_24
        float_53 = rsqrt_24 = None
        float_54 = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_97 = 1.0 + float_54
        float_54 = None
        output_49 = output_48 * add_97
        output_48 = add_97 = None
        hidden_states_73 = output_49.type_as(hidden_states_72)
        output_49 = None
        linear_84 = torch._C._nn.linear(
            hidden_states_73,
            l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_84.view((1, 3, -1, 256))
        linear_84 = None
        query_states_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_73,
            l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_38 = linear_85.view((1, 3, -1, 256))
        linear_85 = None
        key_states_12 = view_38.transpose(1, 2)
        view_38 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_73,
            l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_73 = l_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_39 = linear_86.view((1, 3, -1, 256))
        linear_86 = None
        value_states_12 = view_39.transpose(1, 2)
        view_39 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_114 = query_states_12 * cos_15
        x1_24 = query_states_12[(Ellipsis, slice(None, 128, None))]
        x2_24 = query_states_12[(Ellipsis, slice(128, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_115 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_114 + mul_115
        mul_114 = mul_115 = None
        mul_116 = key_states_12 * cos_15
        cos_15 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 128, None))]
        x2_25 = key_states_12[(Ellipsis, slice(128, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_117 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_116 + mul_117
        mul_116 = mul_117 = None
        getitem_93 = k_embed_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_74 = getitem_93.expand(1, 1, 8, 3, 256)
        getitem_93 = None
        key_24 = hidden_states_74.reshape(1, 8, 3, 256)
        hidden_states_74 = None
        getitem_94 = value_states_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_75 = getitem_94.expand(1, 1, 8, 3, 256)
        getitem_94 = None
        value_24 = hidden_states_75.reshape(1, 8, 3, 256)
        hidden_states_75 = None
        attention_mask_13 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_12 = key_25 = value_25 = attention_mask_13 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_38 = attn_output_49.reshape(1, 3, -1)
        attn_output_49 = None
        attn_output_50 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_76 = hidden_states_72 + attn_output_51
        hidden_states_72 = attn_output_51 = None
        float_55 = hidden_states_76.float()
        pow_26 = float_55.pow(2)
        mean_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_101 = mean_25 + 1e-06
        mean_25 = None
        rsqrt_25 = torch.rsqrt(add_101)
        add_101 = None
        output_50 = float_55 * rsqrt_25
        float_55 = rsqrt_25 = None
        float_56 = (
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_102 = 1.0 + float_56
        float_56 = None
        output_51 = output_50 * add_102
        output_50 = add_102 = None
        hidden_states_77 = output_51.type_as(hidden_states_76)
        output_51 = None
        linear_88 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_12 = torch._C._nn.gelu(linear_88, approximate="tanh")
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_77 = l_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_120 = gelu_12 * linear_89
        gelu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_120,
            l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_120 = l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_78 = hidden_states_76 + down_proj_12
        hidden_states_76 = down_proj_12 = None
        float_57 = hidden_states_78.float()
        pow_27 = float_57.pow(2)
        mean_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_104 = mean_26 + 1e-06
        mean_26 = None
        rsqrt_26 = torch.rsqrt(add_104)
        add_104 = None
        output_52 = float_57 * rsqrt_26
        float_57 = rsqrt_26 = None
        float_58 = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_105 = 1.0 + float_58
        float_58 = None
        output_53 = output_52 * add_105
        output_52 = add_105 = None
        hidden_states_79 = output_53.type_as(hidden_states_78)
        output_53 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_91.view((1, 3, -1, 256))
        linear_91 = None
        query_states_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_41 = linear_92.view((1, 3, -1, 256))
        linear_92 = None
        key_states_13 = view_41.transpose(1, 2)
        view_41 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_79 = l_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_42 = linear_93.view((1, 3, -1, 256))
        linear_93 = None
        value_states_13 = view_42.transpose(1, 2)
        view_42 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_123 = query_states_13 * cos_16
        x1_26 = query_states_13[(Ellipsis, slice(None, 128, None))]
        x2_26 = query_states_13[(Ellipsis, slice(128, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_124 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_123 + mul_124
        mul_123 = mul_124 = None
        mul_125 = key_states_13 * cos_16
        cos_16 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 128, None))]
        x2_27 = key_states_13[(Ellipsis, slice(128, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_126 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_125 + mul_126
        mul_125 = mul_126 = None
        getitem_100 = k_embed_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_80 = getitem_100.expand(1, 1, 8, 3, 256)
        getitem_100 = None
        key_26 = hidden_states_80.reshape(1, 8, 3, 256)
        hidden_states_80 = None
        getitem_101 = value_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_81 = getitem_101.expand(1, 1, 8, 3, 256)
        getitem_101 = None
        value_26 = hidden_states_81.reshape(1, 8, 3, 256)
        hidden_states_81 = None
        attention_mask_14 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_13 = key_27 = value_27 = attention_mask_14 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_41 = attn_output_53.reshape(1, 3, -1)
        attn_output_53 = None
        attn_output_54 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_82 = hidden_states_78 + attn_output_55
        hidden_states_78 = attn_output_55 = None
        float_59 = hidden_states_82.float()
        pow_28 = float_59.pow(2)
        mean_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_109 = mean_27 + 1e-06
        mean_27 = None
        rsqrt_27 = torch.rsqrt(add_109)
        add_109 = None
        output_54 = float_59 * rsqrt_27
        float_59 = rsqrt_27 = None
        float_60 = (
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_110 = 1.0 + float_60
        float_60 = None
        output_55 = output_54 * add_110
        output_54 = add_110 = None
        hidden_states_83 = output_55.type_as(hidden_states_82)
        output_55 = None
        linear_95 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_13 = torch._C._nn.gelu(linear_95, approximate="tanh")
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_83 = l_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_129 = gelu_13 * linear_96
        gelu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_129,
            l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_129 = l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_84 = hidden_states_82 + down_proj_13
        hidden_states_82 = down_proj_13 = None
        float_61 = hidden_states_84.float()
        pow_29 = float_61.pow(2)
        mean_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_112 = mean_28 + 1e-06
        mean_28 = None
        rsqrt_28 = torch.rsqrt(add_112)
        add_112 = None
        output_56 = float_61 * rsqrt_28
        float_61 = rsqrt_28 = None
        float_62 = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_113 = 1.0 + float_62
        float_62 = None
        output_57 = output_56 * add_113
        output_56 = add_113 = None
        hidden_states_85 = output_57.type_as(hidden_states_84)
        output_57 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_98.view((1, 3, -1, 256))
        linear_98 = None
        query_states_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_44 = linear_99.view((1, 3, -1, 256))
        linear_99 = None
        key_states_14 = view_44.transpose(1, 2)
        view_44 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_85 = l_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_45 = linear_100.view((1, 3, -1, 256))
        linear_100 = None
        value_states_14 = view_45.transpose(1, 2)
        view_45 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_132 = query_states_14 * cos_17
        x1_28 = query_states_14[(Ellipsis, slice(None, 128, None))]
        x2_28 = query_states_14[(Ellipsis, slice(128, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_133 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_132 + mul_133
        mul_132 = mul_133 = None
        mul_134 = key_states_14 * cos_17
        cos_17 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 128, None))]
        x2_29 = key_states_14[(Ellipsis, slice(128, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_135 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_134 + mul_135
        mul_134 = mul_135 = None
        getitem_107 = k_embed_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_86 = getitem_107.expand(1, 1, 8, 3, 256)
        getitem_107 = None
        key_28 = hidden_states_86.reshape(1, 8, 3, 256)
        hidden_states_86 = None
        getitem_108 = value_states_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_87 = getitem_108.expand(1, 1, 8, 3, 256)
        getitem_108 = None
        value_28 = hidden_states_87.reshape(1, 8, 3, 256)
        hidden_states_87 = None
        attention_mask_15 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_14 = key_29 = value_29 = attention_mask_15 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_44 = attn_output_57.reshape(1, 3, -1)
        attn_output_57 = None
        attn_output_58 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_88 = hidden_states_84 + attn_output_59
        hidden_states_84 = attn_output_59 = None
        float_63 = hidden_states_88.float()
        pow_30 = float_63.pow(2)
        mean_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_117 = mean_29 + 1e-06
        mean_29 = None
        rsqrt_29 = torch.rsqrt(add_117)
        add_117 = None
        output_58 = float_63 * rsqrt_29
        float_63 = rsqrt_29 = None
        float_64 = (
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_118 = 1.0 + float_64
        float_64 = None
        output_59 = output_58 * add_118
        output_58 = add_118 = None
        hidden_states_89 = output_59.type_as(hidden_states_88)
        output_59 = None
        linear_102 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_14 = torch._C._nn.gelu(linear_102, approximate="tanh")
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_89,
            l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_89 = l_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_138 = gelu_14 * linear_103
        gelu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_138,
            l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_138 = l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_90 = hidden_states_88 + down_proj_14
        hidden_states_88 = down_proj_14 = None
        float_65 = hidden_states_90.float()
        pow_31 = float_65.pow(2)
        mean_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_120 = mean_30 + 1e-06
        mean_30 = None
        rsqrt_30 = torch.rsqrt(add_120)
        add_120 = None
        output_60 = float_65 * rsqrt_30
        float_65 = rsqrt_30 = None
        float_66 = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_121 = 1.0 + float_66
        float_66 = None
        output_61 = output_60 * add_121
        output_60 = add_121 = None
        hidden_states_91 = output_61.type_as(hidden_states_90)
        output_61 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_91,
            l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_105.view((1, 3, -1, 256))
        linear_105 = None
        query_states_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_91,
            l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_47 = linear_106.view((1, 3, -1, 256))
        linear_106 = None
        key_states_15 = view_47.transpose(1, 2)
        view_47 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_91,
            l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_91 = l_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_48 = linear_107.view((1, 3, -1, 256))
        linear_107 = None
        value_states_15 = view_48.transpose(1, 2)
        view_48 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_141 = query_states_15 * cos_18
        x1_30 = query_states_15[(Ellipsis, slice(None, 128, None))]
        x2_30 = query_states_15[(Ellipsis, slice(128, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_142 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_141 + mul_142
        mul_141 = mul_142 = None
        mul_143 = key_states_15 * cos_18
        cos_18 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 128, None))]
        x2_31 = key_states_15[(Ellipsis, slice(128, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_144 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_143 + mul_144
        mul_143 = mul_144 = None
        getitem_114 = k_embed_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_92 = getitem_114.expand(1, 1, 8, 3, 256)
        getitem_114 = None
        key_30 = hidden_states_92.reshape(1, 8, 3, 256)
        hidden_states_92 = None
        getitem_115 = value_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_93 = getitem_115.expand(1, 1, 8, 3, 256)
        getitem_115 = None
        value_30 = hidden_states_93.reshape(1, 8, 3, 256)
        hidden_states_93 = None
        attention_mask_16 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_15 = key_31 = value_31 = attention_mask_16 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_47 = attn_output_61.reshape(1, 3, -1)
        attn_output_61 = None
        attn_output_62 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_94 = hidden_states_90 + attn_output_63
        hidden_states_90 = attn_output_63 = None
        float_67 = hidden_states_94.float()
        pow_32 = float_67.pow(2)
        mean_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_125 = mean_31 + 1e-06
        mean_31 = None
        rsqrt_31 = torch.rsqrt(add_125)
        add_125 = None
        output_62 = float_67 * rsqrt_31
        float_67 = rsqrt_31 = None
        float_68 = (
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_126 = 1.0 + float_68
        float_68 = None
        output_63 = output_62 * add_126
        output_62 = add_126 = None
        hidden_states_95 = output_63.type_as(hidden_states_94)
        output_63 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_15 = torch._C._nn.gelu(linear_109, approximate="tanh")
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_95 = l_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_147 = gelu_15 * linear_110
        gelu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_147,
            l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_147 = l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_96 = hidden_states_94 + down_proj_15
        hidden_states_94 = down_proj_15 = None
        float_69 = hidden_states_96.float()
        pow_33 = float_69.pow(2)
        mean_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_128 = mean_32 + 1e-06
        mean_32 = None
        rsqrt_32 = torch.rsqrt(add_128)
        add_128 = None
        output_64 = float_69 * rsqrt_32
        float_69 = rsqrt_32 = None
        float_70 = (
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_129 = 1.0 + float_70
        float_70 = None
        output_65 = output_64 * add_129
        output_64 = add_129 = None
        hidden_states_97 = output_65.type_as(hidden_states_96)
        output_65 = None
        linear_112 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_112.view((1, 3, -1, 256))
        linear_112 = None
        query_states_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_50 = linear_113.view((1, 3, -1, 256))
        linear_113 = None
        key_states_16 = view_50.transpose(1, 2)
        view_50 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_97 = l_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_51 = linear_114.view((1, 3, -1, 256))
        linear_114 = None
        value_states_16 = view_51.transpose(1, 2)
        view_51 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_150 = query_states_16 * cos_19
        x1_32 = query_states_16[(Ellipsis, slice(None, 128, None))]
        x2_32 = query_states_16[(Ellipsis, slice(128, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_151 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_150 + mul_151
        mul_150 = mul_151 = None
        mul_152 = key_states_16 * cos_19
        cos_19 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 128, None))]
        x2_33 = key_states_16[(Ellipsis, slice(128, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_153 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_152 + mul_153
        mul_152 = mul_153 = None
        getitem_121 = k_embed_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_98 = getitem_121.expand(1, 1, 8, 3, 256)
        getitem_121 = None
        key_32 = hidden_states_98.reshape(1, 8, 3, 256)
        hidden_states_98 = None
        getitem_122 = value_states_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_99 = getitem_122.expand(1, 1, 8, 3, 256)
        getitem_122 = None
        value_32 = hidden_states_99.reshape(1, 8, 3, 256)
        hidden_states_99 = None
        attention_mask_17 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
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
            scale=0.0625,
            is_causal=False,
        )
        query_16 = key_33 = value_33 = attention_mask_17 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_50 = attn_output_65.reshape(1, 3, -1)
        attn_output_65 = None
        attn_output_66 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_100 = hidden_states_96 + attn_output_67
        hidden_states_96 = attn_output_67 = None
        float_71 = hidden_states_100.float()
        pow_34 = float_71.pow(2)
        mean_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_133 = mean_33 + 1e-06
        mean_33 = None
        rsqrt_33 = torch.rsqrt(add_133)
        add_133 = None
        output_66 = float_71 * rsqrt_33
        float_71 = rsqrt_33 = None
        float_72 = (
            l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_134 = 1.0 + float_72
        float_72 = None
        output_67 = output_66 * add_134
        output_66 = add_134 = None
        hidden_states_101 = output_67.type_as(hidden_states_100)
        output_67 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_16 = torch._C._nn.gelu(linear_116, approximate="tanh")
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_101 = l_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_156 = gelu_16 * linear_117
        gelu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_156,
            l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_156 = l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_102 = hidden_states_100 + down_proj_16
        hidden_states_100 = down_proj_16 = None
        float_73 = hidden_states_102.float()
        pow_35 = float_73.pow(2)
        mean_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_136 = mean_34 + 1e-06
        mean_34 = None
        rsqrt_34 = torch.rsqrt(add_136)
        add_136 = None
        output_68 = float_73 * rsqrt_34
        float_73 = rsqrt_34 = None
        float_74 = (
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_137 = 1.0 + float_74
        float_74 = None
        output_69 = output_68 * add_137
        output_68 = add_137 = None
        hidden_states_103 = output_69.type_as(hidden_states_102)
        output_69 = None
        linear_119 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_119.view((1, 3, -1, 256))
        linear_119 = None
        query_states_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_53 = linear_120.view((1, 3, -1, 256))
        linear_120 = None
        key_states_17 = view_53.transpose(1, 2)
        view_53 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_103,
            l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_103 = l_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_54 = linear_121.view((1, 3, -1, 256))
        linear_121 = None
        value_states_17 = view_54.transpose(1, 2)
        view_54 = None
        cos_20 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_20 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_159 = query_states_17 * cos_20
        x1_34 = query_states_17[(Ellipsis, slice(None, 128, None))]
        x2_34 = query_states_17[(Ellipsis, slice(128, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_160 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_159 + mul_160
        mul_159 = mul_160 = None
        mul_161 = key_states_17 * cos_20
        cos_20 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 128, None))]
        x2_35 = key_states_17[(Ellipsis, slice(128, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_162 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_161 + mul_162
        mul_161 = mul_162 = None
        getitem_128 = k_embed_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_104 = getitem_128.expand(1, 1, 8, 3, 256)
        getitem_128 = None
        key_34 = hidden_states_104.reshape(1, 8, 3, 256)
        hidden_states_104 = None
        getitem_129 = value_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        hidden_states_105 = getitem_129.expand(1, 1, 8, 3, 256)
        getitem_129 = None
        value_34 = hidden_states_105.reshape(1, 8, 3, 256)
        hidden_states_105 = None
        attention_mask_18 = causal_mask_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 3, None),
            )
        ]
        causal_mask_2 = None
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
            scale=0.0625,
            is_causal=False,
        )
        query_17 = key_35 = value_35 = attention_mask_18 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_53 = attn_output_69.reshape(1, 3, -1)
        attn_output_69 = None
        attn_output_70 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_106 = hidden_states_102 + attn_output_71
        hidden_states_102 = attn_output_71 = None
        float_75 = hidden_states_106.float()
        pow_36 = float_75.pow(2)
        mean_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_141 = mean_35 + 1e-06
        mean_35 = None
        rsqrt_35 = torch.rsqrt(add_141)
        add_141 = None
        output_70 = float_75 * rsqrt_35
        float_75 = rsqrt_35 = None
        float_76 = (
            l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_142 = 1.0 + float_76
        float_76 = None
        output_71 = output_70 * add_142
        output_70 = add_142 = None
        hidden_states_107 = output_71.type_as(hidden_states_106)
        output_71 = None
        linear_123 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_17 = torch._C._nn.gelu(linear_123, approximate="tanh")
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_107 = l_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_165 = gelu_17 * linear_124
        gelu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_165,
            l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_165 = l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_108 = hidden_states_106 + down_proj_17
        hidden_states_106 = down_proj_17 = None
        float_77 = hidden_states_108.float()
        pow_37 = float_77.pow(2)
        mean_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_144 = mean_36 + 1e-06
        mean_36 = None
        rsqrt_36 = torch.rsqrt(add_144)
        add_144 = None
        output_72 = float_77 * rsqrt_36
        float_77 = rsqrt_36 = None
        float_78 = l_self_modules_norm_parameters_weight_.float()
        l_self_modules_norm_parameters_weight_ = None
        add_145 = 1.0 + float_78
        float_78 = None
        output_73 = output_72 * add_145
        output_72 = add_145 = None
        hidden_states_109 = output_73.type_as(hidden_states_108)
        output_73 = hidden_states_108 = None
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
            hidden_states_109,
        )
