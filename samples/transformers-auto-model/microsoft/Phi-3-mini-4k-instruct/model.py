import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_embeds_: torch.Tensor,
        L_attention_mask_: torch.Tensor,
        L_self_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_qkv_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_mlp_modules_gate_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_2_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_3_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_4_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_5_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_7_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_8_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_9_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_10_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_11_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_12_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_13_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_14_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_15_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_16_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_17_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_18_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_19_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_20_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_21_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_22_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_23_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_24_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_24_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_25_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_25_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_26_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_26_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_27_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_28_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_28_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_29_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_29_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_30_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_30_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = (
            L_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
        )
        l_self_modules_layers_modules_31_modules_self_attn_modules_qkv_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_qkv_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_layers_modules_31_modules_mlp_modules_gate_up_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_mlp_modules_gate_up_proj_parameters_weight_
        l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_
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
        sub = reshaped_cache_position - 2047
        reshaped_cache_position = None
        sliding_mask_overlay = kv_arange_1 > sub
        kv_arange_1 = sub = None
        causal_mask *= sliding_mask_overlay
        causal_mask_1 = causal_mask
        causal_mask = sliding_mask_overlay = None
        getitem_1 = causal_mask_1[
            (None, None, slice(None, None, None), slice(None, None, None))
        ]
        causal_mask_1 = None
        causal_mask_2 = getitem_1.expand(1, -1, -1, -1)
        getitem_1 = None
        getitem_2 = local_padding_mask[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        local_padding_mask = None
        causal_mask_3 = causal_mask_2 * getitem_2
        causal_mask_2 = getitem_2 = None
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
        add = variance + 1e-05
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
        qkv = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states = qkv[(Ellipsis, slice(None, 3072, None))]
        key_states = qkv[(Ellipsis, slice(3072, 6144, None))]
        value_states = qkv[(Ellipsis, slice(6144, None, None))]
        qkv = None
        view_1 = query_states.view((1, 2, -1, 96))
        query_states = None
        query_states_1 = view_1.transpose(1, 2)
        view_1 = None
        view_2 = key_states.view((1, 2, -1, 96))
        key_states = None
        key_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = value_states.view((1, 2, -1, 96))
        value_states = None
        value_states_1 = view_3.transpose(1, 2)
        view_3 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        q_rot = query_states_1[(Ellipsis, slice(None, 96, None))]
        q_pass = query_states_1[(Ellipsis, slice(96, None, None))]
        query_states_1 = None
        k_rot = key_states_1[(Ellipsis, slice(None, 96, None))]
        k_pass = key_states_1[(Ellipsis, slice(96, None, None))]
        key_states_1 = None
        mul_5 = q_rot * cos_3
        x1 = q_rot[(Ellipsis, slice(None, 48, None))]
        x2 = q_rot[(Ellipsis, slice(48, None, None))]
        q_rot = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_6 = cat_1 * sin_3
        cat_1 = None
        add_1 = mul_5 + mul_6
        mul_5 = mul_6 = None
        q_embed = torch.cat([add_1, q_pass], dim=-1)
        add_1 = q_pass = None
        mul_7 = k_rot * cos_3
        cos_3 = None
        x1_1 = k_rot[(Ellipsis, slice(None, 48, None))]
        x2_1 = k_rot[(Ellipsis, slice(48, None, None))]
        k_rot = None
        neg_1 = -x2_1
        x2_1 = None
        cat_3 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_8 = cat_3 * sin_3
        cat_3 = sin_3 = None
        add_2 = mul_7 + mul_8
        mul_7 = mul_8 = None
        k_embed = torch.cat([add_2, k_pass], dim=-1)
        add_2 = k_pass = None
        attention_mask_1 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query = q_embed.contiguous()
        q_embed = None
        key = k_embed.contiguous()
        value = value_states_1.contiguous()
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query = key = value = attention_mask_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape = attn_output_1.reshape(1, 2, -1)
        attn_output_1 = None
        attn_output_2 = reshape.contiguous()
        reshape = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout = torch.nn.functional.dropout(attn_output_3, 0.0, False, False)
        attn_output_3 = None
        hidden_states_3 = l_inputs_embeds_ + dropout
        l_inputs_embeds_ = dropout = None
        hidden_states_4 = hidden_states_3.to(torch.float32)
        pow_2 = hidden_states_4.pow(2)
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_4 = variance_1 + 1e-05
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_4)
        add_4 = None
        hidden_states_5 = hidden_states_4 * rsqrt_1
        hidden_states_4 = rsqrt_1 = None
        to_7 = hidden_states_5.to(torch.bfloat16)
        hidden_states_5 = None
        hidden_states_6 = (
            l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_7
        )
        l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_7
        ) = None
        up_states = torch._C._nn.linear(
            hidden_states_6,
            l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_6 = l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk = up_states.chunk(2, dim=-1)
        up_states = None
        gate = chunk[0]
        up_states_1 = chunk[1]
        chunk = None
        silu = torch.nn.functional.silu(gate, inplace=False)
        gate = None
        up_states_2 = up_states_1 * silu
        up_states_1 = silu = None
        hidden_states_7 = torch._C._nn.linear(
            up_states_2,
            l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_2 = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_1 = torch.nn.functional.dropout(hidden_states_7, 0.0, False, False)
        hidden_states_7 = None
        hidden_states_8 = hidden_states_3 + dropout_1
        hidden_states_3 = dropout_1 = None
        hidden_states_9 = hidden_states_8.to(torch.float32)
        pow_3 = hidden_states_9.pow(2)
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_6 = variance_2 + 1e-05
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_10 = hidden_states_9 * rsqrt_2
        hidden_states_9 = rsqrt_2 = None
        to_9 = hidden_states_10.to(torch.bfloat16)
        hidden_states_10 = None
        hidden_states_11 = (
            l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_9
        ) = None
        qkv_1 = torch._C._nn.linear(
            hidden_states_11,
            l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_11 = l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_2 = qkv_1[(Ellipsis, slice(None, 3072, None))]
        key_states_2 = qkv_1[(Ellipsis, slice(3072, 6144, None))]
        value_states_2 = qkv_1[(Ellipsis, slice(6144, None, None))]
        qkv_1 = None
        view_4 = query_states_2.view((1, 2, -1, 96))
        query_states_2 = None
        query_states_3 = view_4.transpose(1, 2)
        view_4 = None
        view_5 = key_states_2.view((1, 2, -1, 96))
        key_states_2 = None
        key_states_3 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = value_states_2.view((1, 2, -1, 96))
        value_states_2 = None
        value_states_3 = view_6.transpose(1, 2)
        view_6 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        q_rot_1 = query_states_3[(Ellipsis, slice(None, 96, None))]
        q_pass_1 = query_states_3[(Ellipsis, slice(96, None, None))]
        query_states_3 = None
        k_rot_1 = key_states_3[(Ellipsis, slice(None, 96, None))]
        k_pass_1 = key_states_3[(Ellipsis, slice(96, None, None))]
        key_states_3 = None
        mul_14 = q_rot_1 * cos_4
        x1_2 = q_rot_1[(Ellipsis, slice(None, 48, None))]
        x2_2 = q_rot_1[(Ellipsis, slice(48, None, None))]
        q_rot_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_5 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_15 = cat_5 * sin_4
        cat_5 = None
        add_7 = mul_14 + mul_15
        mul_14 = mul_15 = None
        q_embed_1 = torch.cat([add_7, q_pass_1], dim=-1)
        add_7 = q_pass_1 = None
        mul_16 = k_rot_1 * cos_4
        cos_4 = None
        x1_3 = k_rot_1[(Ellipsis, slice(None, 48, None))]
        x2_3 = k_rot_1[(Ellipsis, slice(48, None, None))]
        k_rot_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_7 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_17 = cat_7 * sin_4
        cat_7 = sin_4 = None
        add_8 = mul_16 + mul_17
        mul_16 = mul_17 = None
        k_embed_1 = torch.cat([add_8, k_pass_1], dim=-1)
        add_8 = k_pass_1 = None
        attention_mask_2 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_1 = q_embed_1.contiguous()
        q_embed_1 = None
        key_1 = k_embed_1.contiguous()
        value_1 = value_states_3.contiguous()
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_1 = attn_output_5.reshape(1, 2, -1)
        attn_output_5 = None
        attn_output_6 = reshape_1.contiguous()
        reshape_1 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_2 = torch.nn.functional.dropout(attn_output_7, 0.0, False, False)
        attn_output_7 = None
        hidden_states_12 = hidden_states_8 + dropout_2
        hidden_states_8 = dropout_2 = None
        hidden_states_13 = hidden_states_12.to(torch.float32)
        pow_4 = hidden_states_13.pow(2)
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_10 = variance_3 + 1e-05
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_14 = hidden_states_13 * rsqrt_3
        hidden_states_13 = rsqrt_3 = None
        to_11 = hidden_states_14.to(torch.bfloat16)
        hidden_states_14 = None
        hidden_states_15 = (
            l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_11
        ) = None
        up_states_3 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_15 = l_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_1 = up_states_3.chunk(2, dim=-1)
        up_states_3 = None
        gate_1 = chunk_1[0]
        up_states_4 = chunk_1[1]
        chunk_1 = None
        silu_1 = torch.nn.functional.silu(gate_1, inplace=False)
        gate_1 = None
        up_states_5 = up_states_4 * silu_1
        up_states_4 = silu_1 = None
        hidden_states_16 = torch._C._nn.linear(
            up_states_5,
            l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_5 = l_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_3 = torch.nn.functional.dropout(hidden_states_16, 0.0, False, False)
        hidden_states_16 = None
        hidden_states_17 = hidden_states_12 + dropout_3
        hidden_states_12 = dropout_3 = None
        hidden_states_18 = hidden_states_17.to(torch.float32)
        pow_5 = hidden_states_18.pow(2)
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_12 = variance_4 + 1e-05
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_19 = hidden_states_18 * rsqrt_4
        hidden_states_18 = rsqrt_4 = None
        to_13 = hidden_states_19.to(torch.bfloat16)
        hidden_states_19 = None
        hidden_states_20 = (
            l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_13
        ) = None
        qkv_2 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_layers_modules_2_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_20 = l_self_modules_layers_modules_2_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_4 = qkv_2[(Ellipsis, slice(None, 3072, None))]
        key_states_4 = qkv_2[(Ellipsis, slice(3072, 6144, None))]
        value_states_4 = qkv_2[(Ellipsis, slice(6144, None, None))]
        qkv_2 = None
        view_7 = query_states_4.view((1, 2, -1, 96))
        query_states_4 = None
        query_states_5 = view_7.transpose(1, 2)
        view_7 = None
        view_8 = key_states_4.view((1, 2, -1, 96))
        key_states_4 = None
        key_states_5 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = value_states_4.view((1, 2, -1, 96))
        value_states_4 = None
        value_states_5 = view_9.transpose(1, 2)
        view_9 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        q_rot_2 = query_states_5[(Ellipsis, slice(None, 96, None))]
        q_pass_2 = query_states_5[(Ellipsis, slice(96, None, None))]
        query_states_5 = None
        k_rot_2 = key_states_5[(Ellipsis, slice(None, 96, None))]
        k_pass_2 = key_states_5[(Ellipsis, slice(96, None, None))]
        key_states_5 = None
        mul_23 = q_rot_2 * cos_5
        x1_4 = q_rot_2[(Ellipsis, slice(None, 48, None))]
        x2_4 = q_rot_2[(Ellipsis, slice(48, None, None))]
        q_rot_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_9 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_24 = cat_9 * sin_5
        cat_9 = None
        add_13 = mul_23 + mul_24
        mul_23 = mul_24 = None
        q_embed_2 = torch.cat([add_13, q_pass_2], dim=-1)
        add_13 = q_pass_2 = None
        mul_25 = k_rot_2 * cos_5
        cos_5 = None
        x1_5 = k_rot_2[(Ellipsis, slice(None, 48, None))]
        x2_5 = k_rot_2[(Ellipsis, slice(48, None, None))]
        k_rot_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_11 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_26 = cat_11 * sin_5
        cat_11 = sin_5 = None
        add_14 = mul_25 + mul_26
        mul_25 = mul_26 = None
        k_embed_2 = torch.cat([add_14, k_pass_2], dim=-1)
        add_14 = k_pass_2 = None
        attention_mask_3 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_2 = q_embed_2.contiguous()
        q_embed_2 = None
        key_2 = k_embed_2.contiguous()
        value_2 = value_states_5.contiguous()
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_2 = attn_output_9.reshape(1, 2, -1)
        attn_output_9 = None
        attn_output_10 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_4 = torch.nn.functional.dropout(attn_output_11, 0.0, False, False)
        attn_output_11 = None
        hidden_states_21 = hidden_states_17 + dropout_4
        hidden_states_17 = dropout_4 = None
        hidden_states_22 = hidden_states_21.to(torch.float32)
        pow_6 = hidden_states_22.pow(2)
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_16 = variance_5 + 1e-05
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_23 = hidden_states_22 * rsqrt_5
        hidden_states_22 = rsqrt_5 = None
        to_15 = hidden_states_23.to(torch.bfloat16)
        hidden_states_23 = None
        hidden_states_24 = (
            l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_15
        ) = None
        up_states_6 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_layers_modules_2_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_24 = l_self_modules_layers_modules_2_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_2 = up_states_6.chunk(2, dim=-1)
        up_states_6 = None
        gate_2 = chunk_2[0]
        up_states_7 = chunk_2[1]
        chunk_2 = None
        silu_2 = torch.nn.functional.silu(gate_2, inplace=False)
        gate_2 = None
        up_states_8 = up_states_7 * silu_2
        up_states_7 = silu_2 = None
        hidden_states_25 = torch._C._nn.linear(
            up_states_8,
            l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_8 = l_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_5 = torch.nn.functional.dropout(hidden_states_25, 0.0, False, False)
        hidden_states_25 = None
        hidden_states_26 = hidden_states_21 + dropout_5
        hidden_states_21 = dropout_5 = None
        hidden_states_27 = hidden_states_26.to(torch.float32)
        pow_7 = hidden_states_27.pow(2)
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_18 = variance_6 + 1e-05
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_28 = hidden_states_27 * rsqrt_6
        hidden_states_27 = rsqrt_6 = None
        to_17 = hidden_states_28.to(torch.bfloat16)
        hidden_states_28 = None
        hidden_states_29 = (
            l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_17
        ) = None
        qkv_3 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_layers_modules_3_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_29 = l_self_modules_layers_modules_3_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_6 = qkv_3[(Ellipsis, slice(None, 3072, None))]
        key_states_6 = qkv_3[(Ellipsis, slice(3072, 6144, None))]
        value_states_6 = qkv_3[(Ellipsis, slice(6144, None, None))]
        qkv_3 = None
        view_10 = query_states_6.view((1, 2, -1, 96))
        query_states_6 = None
        query_states_7 = view_10.transpose(1, 2)
        view_10 = None
        view_11 = key_states_6.view((1, 2, -1, 96))
        key_states_6 = None
        key_states_7 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = value_states_6.view((1, 2, -1, 96))
        value_states_6 = None
        value_states_7 = view_12.transpose(1, 2)
        view_12 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        q_rot_3 = query_states_7[(Ellipsis, slice(None, 96, None))]
        q_pass_3 = query_states_7[(Ellipsis, slice(96, None, None))]
        query_states_7 = None
        k_rot_3 = key_states_7[(Ellipsis, slice(None, 96, None))]
        k_pass_3 = key_states_7[(Ellipsis, slice(96, None, None))]
        key_states_7 = None
        mul_32 = q_rot_3 * cos_6
        x1_6 = q_rot_3[(Ellipsis, slice(None, 48, None))]
        x2_6 = q_rot_3[(Ellipsis, slice(48, None, None))]
        q_rot_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_13 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_33 = cat_13 * sin_6
        cat_13 = None
        add_19 = mul_32 + mul_33
        mul_32 = mul_33 = None
        q_embed_3 = torch.cat([add_19, q_pass_3], dim=-1)
        add_19 = q_pass_3 = None
        mul_34 = k_rot_3 * cos_6
        cos_6 = None
        x1_7 = k_rot_3[(Ellipsis, slice(None, 48, None))]
        x2_7 = k_rot_3[(Ellipsis, slice(48, None, None))]
        k_rot_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_15 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_35 = cat_15 * sin_6
        cat_15 = sin_6 = None
        add_20 = mul_34 + mul_35
        mul_34 = mul_35 = None
        k_embed_3 = torch.cat([add_20, k_pass_3], dim=-1)
        add_20 = k_pass_3 = None
        attention_mask_4 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_3 = q_embed_3.contiguous()
        q_embed_3 = None
        key_3 = k_embed_3.contiguous()
        value_3 = value_states_7.contiguous()
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_3 = attn_output_13.reshape(1, 2, -1)
        attn_output_13 = None
        attn_output_14 = reshape_3.contiguous()
        reshape_3 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_6 = torch.nn.functional.dropout(attn_output_15, 0.0, False, False)
        attn_output_15 = None
        hidden_states_30 = hidden_states_26 + dropout_6
        hidden_states_26 = dropout_6 = None
        hidden_states_31 = hidden_states_30.to(torch.float32)
        pow_8 = hidden_states_31.pow(2)
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_22 = variance_7 + 1e-05
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_32 = hidden_states_31 * rsqrt_7
        hidden_states_31 = rsqrt_7 = None
        to_19 = hidden_states_32.to(torch.bfloat16)
        hidden_states_32 = None
        hidden_states_33 = (
            l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_19
        ) = None
        up_states_9 = torch._C._nn.linear(
            hidden_states_33,
            l_self_modules_layers_modules_3_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_33 = l_self_modules_layers_modules_3_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_3 = up_states_9.chunk(2, dim=-1)
        up_states_9 = None
        gate_3 = chunk_3[0]
        up_states_10 = chunk_3[1]
        chunk_3 = None
        silu_3 = torch.nn.functional.silu(gate_3, inplace=False)
        gate_3 = None
        up_states_11 = up_states_10 * silu_3
        up_states_10 = silu_3 = None
        hidden_states_34 = torch._C._nn.linear(
            up_states_11,
            l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_11 = l_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_7 = torch.nn.functional.dropout(hidden_states_34, 0.0, False, False)
        hidden_states_34 = None
        hidden_states_35 = hidden_states_30 + dropout_7
        hidden_states_30 = dropout_7 = None
        hidden_states_36 = hidden_states_35.to(torch.float32)
        pow_9 = hidden_states_36.pow(2)
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_24 = variance_8 + 1e-05
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_37 = hidden_states_36 * rsqrt_8
        hidden_states_36 = rsqrt_8 = None
        to_21 = hidden_states_37.to(torch.bfloat16)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_21
        ) = None
        qkv_4 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_layers_modules_4_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_layers_modules_4_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_8 = qkv_4[(Ellipsis, slice(None, 3072, None))]
        key_states_8 = qkv_4[(Ellipsis, slice(3072, 6144, None))]
        value_states_8 = qkv_4[(Ellipsis, slice(6144, None, None))]
        qkv_4 = None
        view_13 = query_states_8.view((1, 2, -1, 96))
        query_states_8 = None
        query_states_9 = view_13.transpose(1, 2)
        view_13 = None
        view_14 = key_states_8.view((1, 2, -1, 96))
        key_states_8 = None
        key_states_9 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = value_states_8.view((1, 2, -1, 96))
        value_states_8 = None
        value_states_9 = view_15.transpose(1, 2)
        view_15 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        q_rot_4 = query_states_9[(Ellipsis, slice(None, 96, None))]
        q_pass_4 = query_states_9[(Ellipsis, slice(96, None, None))]
        query_states_9 = None
        k_rot_4 = key_states_9[(Ellipsis, slice(None, 96, None))]
        k_pass_4 = key_states_9[(Ellipsis, slice(96, None, None))]
        key_states_9 = None
        mul_41 = q_rot_4 * cos_7
        x1_8 = q_rot_4[(Ellipsis, slice(None, 48, None))]
        x2_8 = q_rot_4[(Ellipsis, slice(48, None, None))]
        q_rot_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_17 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_42 = cat_17 * sin_7
        cat_17 = None
        add_25 = mul_41 + mul_42
        mul_41 = mul_42 = None
        q_embed_4 = torch.cat([add_25, q_pass_4], dim=-1)
        add_25 = q_pass_4 = None
        mul_43 = k_rot_4 * cos_7
        cos_7 = None
        x1_9 = k_rot_4[(Ellipsis, slice(None, 48, None))]
        x2_9 = k_rot_4[(Ellipsis, slice(48, None, None))]
        k_rot_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_19 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_44 = cat_19 * sin_7
        cat_19 = sin_7 = None
        add_26 = mul_43 + mul_44
        mul_43 = mul_44 = None
        k_embed_4 = torch.cat([add_26, k_pass_4], dim=-1)
        add_26 = k_pass_4 = None
        attention_mask_5 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_4 = q_embed_4.contiguous()
        q_embed_4 = None
        key_4 = k_embed_4.contiguous()
        value_4 = value_states_9.contiguous()
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_4 = attn_output_17.reshape(1, 2, -1)
        attn_output_17 = None
        attn_output_18 = reshape_4.contiguous()
        reshape_4 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_8 = torch.nn.functional.dropout(attn_output_19, 0.0, False, False)
        attn_output_19 = None
        hidden_states_39 = hidden_states_35 + dropout_8
        hidden_states_35 = dropout_8 = None
        hidden_states_40 = hidden_states_39.to(torch.float32)
        pow_10 = hidden_states_40.pow(2)
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_28 = variance_9 + 1e-05
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_41 = hidden_states_40 * rsqrt_9
        hidden_states_40 = rsqrt_9 = None
        to_23 = hidden_states_41.to(torch.bfloat16)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_23
        ) = None
        up_states_12 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_layers_modules_4_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_layers_modules_4_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_4 = up_states_12.chunk(2, dim=-1)
        up_states_12 = None
        gate_4 = chunk_4[0]
        up_states_13 = chunk_4[1]
        chunk_4 = None
        silu_4 = torch.nn.functional.silu(gate_4, inplace=False)
        gate_4 = None
        up_states_14 = up_states_13 * silu_4
        up_states_13 = silu_4 = None
        hidden_states_43 = torch._C._nn.linear(
            up_states_14,
            l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_14 = l_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_9 = torch.nn.functional.dropout(hidden_states_43, 0.0, False, False)
        hidden_states_43 = None
        hidden_states_44 = hidden_states_39 + dropout_9
        hidden_states_39 = dropout_9 = None
        hidden_states_45 = hidden_states_44.to(torch.float32)
        pow_11 = hidden_states_45.pow(2)
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_30 = variance_10 + 1e-05
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_46 = hidden_states_45 * rsqrt_10
        hidden_states_45 = rsqrt_10 = None
        to_25 = hidden_states_46.to(torch.bfloat16)
        hidden_states_46 = None
        hidden_states_47 = (
            l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_25
        ) = None
        qkv_5 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_layers_modules_5_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_layers_modules_5_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_10 = qkv_5[(Ellipsis, slice(None, 3072, None))]
        key_states_10 = qkv_5[(Ellipsis, slice(3072, 6144, None))]
        value_states_10 = qkv_5[(Ellipsis, slice(6144, None, None))]
        qkv_5 = None
        view_16 = query_states_10.view((1, 2, -1, 96))
        query_states_10 = None
        query_states_11 = view_16.transpose(1, 2)
        view_16 = None
        view_17 = key_states_10.view((1, 2, -1, 96))
        key_states_10 = None
        key_states_11 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = value_states_10.view((1, 2, -1, 96))
        value_states_10 = None
        value_states_11 = view_18.transpose(1, 2)
        view_18 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        q_rot_5 = query_states_11[(Ellipsis, slice(None, 96, None))]
        q_pass_5 = query_states_11[(Ellipsis, slice(96, None, None))]
        query_states_11 = None
        k_rot_5 = key_states_11[(Ellipsis, slice(None, 96, None))]
        k_pass_5 = key_states_11[(Ellipsis, slice(96, None, None))]
        key_states_11 = None
        mul_50 = q_rot_5 * cos_8
        x1_10 = q_rot_5[(Ellipsis, slice(None, 48, None))]
        x2_10 = q_rot_5[(Ellipsis, slice(48, None, None))]
        q_rot_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_21 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_51 = cat_21 * sin_8
        cat_21 = None
        add_31 = mul_50 + mul_51
        mul_50 = mul_51 = None
        q_embed_5 = torch.cat([add_31, q_pass_5], dim=-1)
        add_31 = q_pass_5 = None
        mul_52 = k_rot_5 * cos_8
        cos_8 = None
        x1_11 = k_rot_5[(Ellipsis, slice(None, 48, None))]
        x2_11 = k_rot_5[(Ellipsis, slice(48, None, None))]
        k_rot_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_23 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_53 = cat_23 * sin_8
        cat_23 = sin_8 = None
        add_32 = mul_52 + mul_53
        mul_52 = mul_53 = None
        k_embed_5 = torch.cat([add_32, k_pass_5], dim=-1)
        add_32 = k_pass_5 = None
        attention_mask_6 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_5 = q_embed_5.contiguous()
        q_embed_5 = None
        key_5 = k_embed_5.contiguous()
        value_5 = value_states_11.contiguous()
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_5 = attn_output_21.reshape(1, 2, -1)
        attn_output_21 = None
        attn_output_22 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_10 = torch.nn.functional.dropout(attn_output_23, 0.0, False, False)
        attn_output_23 = None
        hidden_states_48 = hidden_states_44 + dropout_10
        hidden_states_44 = dropout_10 = None
        hidden_states_49 = hidden_states_48.to(torch.float32)
        pow_12 = hidden_states_49.pow(2)
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_34 = variance_11 + 1e-05
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_50 = hidden_states_49 * rsqrt_11
        hidden_states_49 = rsqrt_11 = None
        to_27 = hidden_states_50.to(torch.bfloat16)
        hidden_states_50 = None
        hidden_states_51 = (
            l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_27
        ) = None
        up_states_15 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_layers_modules_5_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_51 = l_self_modules_layers_modules_5_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_5 = up_states_15.chunk(2, dim=-1)
        up_states_15 = None
        gate_5 = chunk_5[0]
        up_states_16 = chunk_5[1]
        chunk_5 = None
        silu_5 = torch.nn.functional.silu(gate_5, inplace=False)
        gate_5 = None
        up_states_17 = up_states_16 * silu_5
        up_states_16 = silu_5 = None
        hidden_states_52 = torch._C._nn.linear(
            up_states_17,
            l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_17 = l_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_11 = torch.nn.functional.dropout(hidden_states_52, 0.0, False, False)
        hidden_states_52 = None
        hidden_states_53 = hidden_states_48 + dropout_11
        hidden_states_48 = dropout_11 = None
        hidden_states_54 = hidden_states_53.to(torch.float32)
        pow_13 = hidden_states_54.pow(2)
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_36 = variance_12 + 1e-05
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_55 = hidden_states_54 * rsqrt_12
        hidden_states_54 = rsqrt_12 = None
        to_29 = hidden_states_55.to(torch.bfloat16)
        hidden_states_55 = None
        hidden_states_56 = (
            l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_29
        ) = None
        qkv_6 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_56 = l_self_modules_layers_modules_6_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_12 = qkv_6[(Ellipsis, slice(None, 3072, None))]
        key_states_12 = qkv_6[(Ellipsis, slice(3072, 6144, None))]
        value_states_12 = qkv_6[(Ellipsis, slice(6144, None, None))]
        qkv_6 = None
        view_19 = query_states_12.view((1, 2, -1, 96))
        query_states_12 = None
        query_states_13 = view_19.transpose(1, 2)
        view_19 = None
        view_20 = key_states_12.view((1, 2, -1, 96))
        key_states_12 = None
        key_states_13 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = value_states_12.view((1, 2, -1, 96))
        value_states_12 = None
        value_states_13 = view_21.transpose(1, 2)
        view_21 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        q_rot_6 = query_states_13[(Ellipsis, slice(None, 96, None))]
        q_pass_6 = query_states_13[(Ellipsis, slice(96, None, None))]
        query_states_13 = None
        k_rot_6 = key_states_13[(Ellipsis, slice(None, 96, None))]
        k_pass_6 = key_states_13[(Ellipsis, slice(96, None, None))]
        key_states_13 = None
        mul_59 = q_rot_6 * cos_9
        x1_12 = q_rot_6[(Ellipsis, slice(None, 48, None))]
        x2_12 = q_rot_6[(Ellipsis, slice(48, None, None))]
        q_rot_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_25 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_60 = cat_25 * sin_9
        cat_25 = None
        add_37 = mul_59 + mul_60
        mul_59 = mul_60 = None
        q_embed_6 = torch.cat([add_37, q_pass_6], dim=-1)
        add_37 = q_pass_6 = None
        mul_61 = k_rot_6 * cos_9
        cos_9 = None
        x1_13 = k_rot_6[(Ellipsis, slice(None, 48, None))]
        x2_13 = k_rot_6[(Ellipsis, slice(48, None, None))]
        k_rot_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_27 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_62 = cat_27 * sin_9
        cat_27 = sin_9 = None
        add_38 = mul_61 + mul_62
        mul_61 = mul_62 = None
        k_embed_6 = torch.cat([add_38, k_pass_6], dim=-1)
        add_38 = k_pass_6 = None
        attention_mask_7 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_6 = q_embed_6.contiguous()
        q_embed_6 = None
        key_6 = k_embed_6.contiguous()
        value_6 = value_states_13.contiguous()
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_6 = key_6 = value_6 = attention_mask_7 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_6 = attn_output_25.reshape(1, 2, -1)
        attn_output_25 = None
        attn_output_26 = reshape_6.contiguous()
        reshape_6 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_12 = torch.nn.functional.dropout(attn_output_27, 0.0, False, False)
        attn_output_27 = None
        hidden_states_57 = hidden_states_53 + dropout_12
        hidden_states_53 = dropout_12 = None
        hidden_states_58 = hidden_states_57.to(torch.float32)
        pow_14 = hidden_states_58.pow(2)
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_40 = variance_13 + 1e-05
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_59 = hidden_states_58 * rsqrt_13
        hidden_states_58 = rsqrt_13 = None
        to_31 = hidden_states_59.to(torch.bfloat16)
        hidden_states_59 = None
        hidden_states_60 = (
            l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_31
        ) = None
        up_states_18 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_layers_modules_6_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_60 = l_self_modules_layers_modules_6_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_6 = up_states_18.chunk(2, dim=-1)
        up_states_18 = None
        gate_6 = chunk_6[0]
        up_states_19 = chunk_6[1]
        chunk_6 = None
        silu_6 = torch.nn.functional.silu(gate_6, inplace=False)
        gate_6 = None
        up_states_20 = up_states_19 * silu_6
        up_states_19 = silu_6 = None
        hidden_states_61 = torch._C._nn.linear(
            up_states_20,
            l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_20 = l_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_13 = torch.nn.functional.dropout(hidden_states_61, 0.0, False, False)
        hidden_states_61 = None
        hidden_states_62 = hidden_states_57 + dropout_13
        hidden_states_57 = dropout_13 = None
        hidden_states_63 = hidden_states_62.to(torch.float32)
        pow_15 = hidden_states_63.pow(2)
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_42 = variance_14 + 1e-05
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_64 = hidden_states_63 * rsqrt_14
        hidden_states_63 = rsqrt_14 = None
        to_33 = hidden_states_64.to(torch.bfloat16)
        hidden_states_64 = None
        hidden_states_65 = (
            l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_33
        ) = None
        qkv_7 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_layers_modules_7_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_65 = l_self_modules_layers_modules_7_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_14 = qkv_7[(Ellipsis, slice(None, 3072, None))]
        key_states_14 = qkv_7[(Ellipsis, slice(3072, 6144, None))]
        value_states_14 = qkv_7[(Ellipsis, slice(6144, None, None))]
        qkv_7 = None
        view_22 = query_states_14.view((1, 2, -1, 96))
        query_states_14 = None
        query_states_15 = view_22.transpose(1, 2)
        view_22 = None
        view_23 = key_states_14.view((1, 2, -1, 96))
        key_states_14 = None
        key_states_15 = view_23.transpose(1, 2)
        view_23 = None
        view_24 = value_states_14.view((1, 2, -1, 96))
        value_states_14 = None
        value_states_15 = view_24.transpose(1, 2)
        view_24 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        q_rot_7 = query_states_15[(Ellipsis, slice(None, 96, None))]
        q_pass_7 = query_states_15[(Ellipsis, slice(96, None, None))]
        query_states_15 = None
        k_rot_7 = key_states_15[(Ellipsis, slice(None, 96, None))]
        k_pass_7 = key_states_15[(Ellipsis, slice(96, None, None))]
        key_states_15 = None
        mul_68 = q_rot_7 * cos_10
        x1_14 = q_rot_7[(Ellipsis, slice(None, 48, None))]
        x2_14 = q_rot_7[(Ellipsis, slice(48, None, None))]
        q_rot_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_29 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_69 = cat_29 * sin_10
        cat_29 = None
        add_43 = mul_68 + mul_69
        mul_68 = mul_69 = None
        q_embed_7 = torch.cat([add_43, q_pass_7], dim=-1)
        add_43 = q_pass_7 = None
        mul_70 = k_rot_7 * cos_10
        cos_10 = None
        x1_15 = k_rot_7[(Ellipsis, slice(None, 48, None))]
        x2_15 = k_rot_7[(Ellipsis, slice(48, None, None))]
        k_rot_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_31 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_71 = cat_31 * sin_10
        cat_31 = sin_10 = None
        add_44 = mul_70 + mul_71
        mul_70 = mul_71 = None
        k_embed_7 = torch.cat([add_44, k_pass_7], dim=-1)
        add_44 = k_pass_7 = None
        attention_mask_8 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_7 = q_embed_7.contiguous()
        q_embed_7 = None
        key_7 = k_embed_7.contiguous()
        value_7 = value_states_15.contiguous()
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_7 = key_7 = value_7 = attention_mask_8 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_7 = attn_output_29.reshape(1, 2, -1)
        attn_output_29 = None
        attn_output_30 = reshape_7.contiguous()
        reshape_7 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_14 = torch.nn.functional.dropout(attn_output_31, 0.0, False, False)
        attn_output_31 = None
        hidden_states_66 = hidden_states_62 + dropout_14
        hidden_states_62 = dropout_14 = None
        hidden_states_67 = hidden_states_66.to(torch.float32)
        pow_16 = hidden_states_67.pow(2)
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_46 = variance_15 + 1e-05
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_68 = hidden_states_67 * rsqrt_15
        hidden_states_67 = rsqrt_15 = None
        to_35 = hidden_states_68.to(torch.bfloat16)
        hidden_states_68 = None
        hidden_states_69 = (
            l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_35
        ) = None
        up_states_21 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_layers_modules_7_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_69 = l_self_modules_layers_modules_7_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_7 = up_states_21.chunk(2, dim=-1)
        up_states_21 = None
        gate_7 = chunk_7[0]
        up_states_22 = chunk_7[1]
        chunk_7 = None
        silu_7 = torch.nn.functional.silu(gate_7, inplace=False)
        gate_7 = None
        up_states_23 = up_states_22 * silu_7
        up_states_22 = silu_7 = None
        hidden_states_70 = torch._C._nn.linear(
            up_states_23,
            l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_23 = l_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_15 = torch.nn.functional.dropout(hidden_states_70, 0.0, False, False)
        hidden_states_70 = None
        hidden_states_71 = hidden_states_66 + dropout_15
        hidden_states_66 = dropout_15 = None
        hidden_states_72 = hidden_states_71.to(torch.float32)
        pow_17 = hidden_states_72.pow(2)
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_48 = variance_16 + 1e-05
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_73 = hidden_states_72 * rsqrt_16
        hidden_states_72 = rsqrt_16 = None
        to_37 = hidden_states_73.to(torch.bfloat16)
        hidden_states_73 = None
        hidden_states_74 = (
            l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_37
        ) = None
        qkv_8 = torch._C._nn.linear(
            hidden_states_74,
            l_self_modules_layers_modules_8_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_74 = l_self_modules_layers_modules_8_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_16 = qkv_8[(Ellipsis, slice(None, 3072, None))]
        key_states_16 = qkv_8[(Ellipsis, slice(3072, 6144, None))]
        value_states_16 = qkv_8[(Ellipsis, slice(6144, None, None))]
        qkv_8 = None
        view_25 = query_states_16.view((1, 2, -1, 96))
        query_states_16 = None
        query_states_17 = view_25.transpose(1, 2)
        view_25 = None
        view_26 = key_states_16.view((1, 2, -1, 96))
        key_states_16 = None
        key_states_17 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = value_states_16.view((1, 2, -1, 96))
        value_states_16 = None
        value_states_17 = view_27.transpose(1, 2)
        view_27 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        q_rot_8 = query_states_17[(Ellipsis, slice(None, 96, None))]
        q_pass_8 = query_states_17[(Ellipsis, slice(96, None, None))]
        query_states_17 = None
        k_rot_8 = key_states_17[(Ellipsis, slice(None, 96, None))]
        k_pass_8 = key_states_17[(Ellipsis, slice(96, None, None))]
        key_states_17 = None
        mul_77 = q_rot_8 * cos_11
        x1_16 = q_rot_8[(Ellipsis, slice(None, 48, None))]
        x2_16 = q_rot_8[(Ellipsis, slice(48, None, None))]
        q_rot_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_33 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_78 = cat_33 * sin_11
        cat_33 = None
        add_49 = mul_77 + mul_78
        mul_77 = mul_78 = None
        q_embed_8 = torch.cat([add_49, q_pass_8], dim=-1)
        add_49 = q_pass_8 = None
        mul_79 = k_rot_8 * cos_11
        cos_11 = None
        x1_17 = k_rot_8[(Ellipsis, slice(None, 48, None))]
        x2_17 = k_rot_8[(Ellipsis, slice(48, None, None))]
        k_rot_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_35 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_80 = cat_35 * sin_11
        cat_35 = sin_11 = None
        add_50 = mul_79 + mul_80
        mul_79 = mul_80 = None
        k_embed_8 = torch.cat([add_50, k_pass_8], dim=-1)
        add_50 = k_pass_8 = None
        attention_mask_9 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_8 = q_embed_8.contiguous()
        q_embed_8 = None
        key_8 = k_embed_8.contiguous()
        value_8 = value_states_17.contiguous()
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_8 = key_8 = value_8 = attention_mask_9 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_8 = attn_output_33.reshape(1, 2, -1)
        attn_output_33 = None
        attn_output_34 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_16 = torch.nn.functional.dropout(attn_output_35, 0.0, False, False)
        attn_output_35 = None
        hidden_states_75 = hidden_states_71 + dropout_16
        hidden_states_71 = dropout_16 = None
        hidden_states_76 = hidden_states_75.to(torch.float32)
        pow_18 = hidden_states_76.pow(2)
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_52 = variance_17 + 1e-05
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_77 = hidden_states_76 * rsqrt_17
        hidden_states_76 = rsqrt_17 = None
        to_39 = hidden_states_77.to(torch.bfloat16)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_39
        ) = None
        up_states_24 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_layers_modules_8_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_78 = l_self_modules_layers_modules_8_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_8 = up_states_24.chunk(2, dim=-1)
        up_states_24 = None
        gate_8 = chunk_8[0]
        up_states_25 = chunk_8[1]
        chunk_8 = None
        silu_8 = torch.nn.functional.silu(gate_8, inplace=False)
        gate_8 = None
        up_states_26 = up_states_25 * silu_8
        up_states_25 = silu_8 = None
        hidden_states_79 = torch._C._nn.linear(
            up_states_26,
            l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_26 = l_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_17 = torch.nn.functional.dropout(hidden_states_79, 0.0, False, False)
        hidden_states_79 = None
        hidden_states_80 = hidden_states_75 + dropout_17
        hidden_states_75 = dropout_17 = None
        hidden_states_81 = hidden_states_80.to(torch.float32)
        pow_19 = hidden_states_81.pow(2)
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_54 = variance_18 + 1e-05
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_82 = hidden_states_81 * rsqrt_18
        hidden_states_81 = rsqrt_18 = None
        to_41 = hidden_states_82.to(torch.bfloat16)
        hidden_states_82 = None
        hidden_states_83 = (
            l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_41
        )
        l_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_41
        ) = None
        qkv_9 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_layers_modules_9_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_83 = l_self_modules_layers_modules_9_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_18 = qkv_9[(Ellipsis, slice(None, 3072, None))]
        key_states_18 = qkv_9[(Ellipsis, slice(3072, 6144, None))]
        value_states_18 = qkv_9[(Ellipsis, slice(6144, None, None))]
        qkv_9 = None
        view_28 = query_states_18.view((1, 2, -1, 96))
        query_states_18 = None
        query_states_19 = view_28.transpose(1, 2)
        view_28 = None
        view_29 = key_states_18.view((1, 2, -1, 96))
        key_states_18 = None
        key_states_19 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = value_states_18.view((1, 2, -1, 96))
        value_states_18 = None
        value_states_19 = view_30.transpose(1, 2)
        view_30 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        q_rot_9 = query_states_19[(Ellipsis, slice(None, 96, None))]
        q_pass_9 = query_states_19[(Ellipsis, slice(96, None, None))]
        query_states_19 = None
        k_rot_9 = key_states_19[(Ellipsis, slice(None, 96, None))]
        k_pass_9 = key_states_19[(Ellipsis, slice(96, None, None))]
        key_states_19 = None
        mul_86 = q_rot_9 * cos_12
        x1_18 = q_rot_9[(Ellipsis, slice(None, 48, None))]
        x2_18 = q_rot_9[(Ellipsis, slice(48, None, None))]
        q_rot_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_37 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_87 = cat_37 * sin_12
        cat_37 = None
        add_55 = mul_86 + mul_87
        mul_86 = mul_87 = None
        q_embed_9 = torch.cat([add_55, q_pass_9], dim=-1)
        add_55 = q_pass_9 = None
        mul_88 = k_rot_9 * cos_12
        cos_12 = None
        x1_19 = k_rot_9[(Ellipsis, slice(None, 48, None))]
        x2_19 = k_rot_9[(Ellipsis, slice(48, None, None))]
        k_rot_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_39 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_89 = cat_39 * sin_12
        cat_39 = sin_12 = None
        add_56 = mul_88 + mul_89
        mul_88 = mul_89 = None
        k_embed_9 = torch.cat([add_56, k_pass_9], dim=-1)
        add_56 = k_pass_9 = None
        attention_mask_10 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_9 = q_embed_9.contiguous()
        q_embed_9 = None
        key_9 = k_embed_9.contiguous()
        value_9 = value_states_19.contiguous()
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_9 = key_9 = value_9 = attention_mask_10 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_9 = attn_output_37.reshape(1, 2, -1)
        attn_output_37 = None
        attn_output_38 = reshape_9.contiguous()
        reshape_9 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_18 = torch.nn.functional.dropout(attn_output_39, 0.0, False, False)
        attn_output_39 = None
        hidden_states_84 = hidden_states_80 + dropout_18
        hidden_states_80 = dropout_18 = None
        hidden_states_85 = hidden_states_84.to(torch.float32)
        pow_20 = hidden_states_85.pow(2)
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_58 = variance_19 + 1e-05
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_58)
        add_58 = None
        hidden_states_86 = hidden_states_85 * rsqrt_19
        hidden_states_85 = rsqrt_19 = None
        to_43 = hidden_states_86.to(torch.bfloat16)
        hidden_states_86 = None
        hidden_states_87 = (
            l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_43
        ) = None
        up_states_27 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_layers_modules_9_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_87 = l_self_modules_layers_modules_9_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_9 = up_states_27.chunk(2, dim=-1)
        up_states_27 = None
        gate_9 = chunk_9[0]
        up_states_28 = chunk_9[1]
        chunk_9 = None
        silu_9 = torch.nn.functional.silu(gate_9, inplace=False)
        gate_9 = None
        up_states_29 = up_states_28 * silu_9
        up_states_28 = silu_9 = None
        hidden_states_88 = torch._C._nn.linear(
            up_states_29,
            l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_29 = l_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_19 = torch.nn.functional.dropout(hidden_states_88, 0.0, False, False)
        hidden_states_88 = None
        hidden_states_89 = hidden_states_84 + dropout_19
        hidden_states_84 = dropout_19 = None
        hidden_states_90 = hidden_states_89.to(torch.float32)
        pow_21 = hidden_states_90.pow(2)
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_60 = variance_20 + 1e-05
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_60)
        add_60 = None
        hidden_states_91 = hidden_states_90 * rsqrt_20
        hidden_states_90 = rsqrt_20 = None
        to_45 = hidden_states_91.to(torch.bfloat16)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_45
        ) = None
        qkv_10 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_layers_modules_10_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_92 = l_self_modules_layers_modules_10_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_20 = qkv_10[(Ellipsis, slice(None, 3072, None))]
        key_states_20 = qkv_10[(Ellipsis, slice(3072, 6144, None))]
        value_states_20 = qkv_10[(Ellipsis, slice(6144, None, None))]
        qkv_10 = None
        view_31 = query_states_20.view((1, 2, -1, 96))
        query_states_20 = None
        query_states_21 = view_31.transpose(1, 2)
        view_31 = None
        view_32 = key_states_20.view((1, 2, -1, 96))
        key_states_20 = None
        key_states_21 = view_32.transpose(1, 2)
        view_32 = None
        view_33 = value_states_20.view((1, 2, -1, 96))
        value_states_20 = None
        value_states_21 = view_33.transpose(1, 2)
        view_33 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        q_rot_10 = query_states_21[(Ellipsis, slice(None, 96, None))]
        q_pass_10 = query_states_21[(Ellipsis, slice(96, None, None))]
        query_states_21 = None
        k_rot_10 = key_states_21[(Ellipsis, slice(None, 96, None))]
        k_pass_10 = key_states_21[(Ellipsis, slice(96, None, None))]
        key_states_21 = None
        mul_95 = q_rot_10 * cos_13
        x1_20 = q_rot_10[(Ellipsis, slice(None, 48, None))]
        x2_20 = q_rot_10[(Ellipsis, slice(48, None, None))]
        q_rot_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_41 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_96 = cat_41 * sin_13
        cat_41 = None
        add_61 = mul_95 + mul_96
        mul_95 = mul_96 = None
        q_embed_10 = torch.cat([add_61, q_pass_10], dim=-1)
        add_61 = q_pass_10 = None
        mul_97 = k_rot_10 * cos_13
        cos_13 = None
        x1_21 = k_rot_10[(Ellipsis, slice(None, 48, None))]
        x2_21 = k_rot_10[(Ellipsis, slice(48, None, None))]
        k_rot_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_43 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_98 = cat_43 * sin_13
        cat_43 = sin_13 = None
        add_62 = mul_97 + mul_98
        mul_97 = mul_98 = None
        k_embed_10 = torch.cat([add_62, k_pass_10], dim=-1)
        add_62 = k_pass_10 = None
        attention_mask_11 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_10 = q_embed_10.contiguous()
        q_embed_10 = None
        key_10 = k_embed_10.contiguous()
        value_10 = value_states_21.contiguous()
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_10 = key_10 = value_10 = attention_mask_11 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_10 = attn_output_41.reshape(1, 2, -1)
        attn_output_41 = None
        attn_output_42 = reshape_10.contiguous()
        reshape_10 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_20 = torch.nn.functional.dropout(attn_output_43, 0.0, False, False)
        attn_output_43 = None
        hidden_states_93 = hidden_states_89 + dropout_20
        hidden_states_89 = dropout_20 = None
        hidden_states_94 = hidden_states_93.to(torch.float32)
        pow_22 = hidden_states_94.pow(2)
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_64 = variance_21 + 1e-05
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_95 = hidden_states_94 * rsqrt_21
        hidden_states_94 = rsqrt_21 = None
        to_47 = hidden_states_95.to(torch.bfloat16)
        hidden_states_95 = None
        hidden_states_96 = (
            l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_47
        )
        l_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_47
        ) = None
        up_states_30 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_layers_modules_10_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_96 = l_self_modules_layers_modules_10_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_10 = up_states_30.chunk(2, dim=-1)
        up_states_30 = None
        gate_10 = chunk_10[0]
        up_states_31 = chunk_10[1]
        chunk_10 = None
        silu_10 = torch.nn.functional.silu(gate_10, inplace=False)
        gate_10 = None
        up_states_32 = up_states_31 * silu_10
        up_states_31 = silu_10 = None
        hidden_states_97 = torch._C._nn.linear(
            up_states_32,
            l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_32 = l_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_21 = torch.nn.functional.dropout(hidden_states_97, 0.0, False, False)
        hidden_states_97 = None
        hidden_states_98 = hidden_states_93 + dropout_21
        hidden_states_93 = dropout_21 = None
        hidden_states_99 = hidden_states_98.to(torch.float32)
        pow_23 = hidden_states_99.pow(2)
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_66 = variance_22 + 1e-05
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_100 = hidden_states_99 * rsqrt_22
        hidden_states_99 = rsqrt_22 = None
        to_49 = hidden_states_100.to(torch.bfloat16)
        hidden_states_100 = None
        hidden_states_101 = (
            l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_49
        ) = None
        qkv_11 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_layers_modules_11_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_101 = l_self_modules_layers_modules_11_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_22 = qkv_11[(Ellipsis, slice(None, 3072, None))]
        key_states_22 = qkv_11[(Ellipsis, slice(3072, 6144, None))]
        value_states_22 = qkv_11[(Ellipsis, slice(6144, None, None))]
        qkv_11 = None
        view_34 = query_states_22.view((1, 2, -1, 96))
        query_states_22 = None
        query_states_23 = view_34.transpose(1, 2)
        view_34 = None
        view_35 = key_states_22.view((1, 2, -1, 96))
        key_states_22 = None
        key_states_23 = view_35.transpose(1, 2)
        view_35 = None
        view_36 = value_states_22.view((1, 2, -1, 96))
        value_states_22 = None
        value_states_23 = view_36.transpose(1, 2)
        view_36 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        q_rot_11 = query_states_23[(Ellipsis, slice(None, 96, None))]
        q_pass_11 = query_states_23[(Ellipsis, slice(96, None, None))]
        query_states_23 = None
        k_rot_11 = key_states_23[(Ellipsis, slice(None, 96, None))]
        k_pass_11 = key_states_23[(Ellipsis, slice(96, None, None))]
        key_states_23 = None
        mul_104 = q_rot_11 * cos_14
        x1_22 = q_rot_11[(Ellipsis, slice(None, 48, None))]
        x2_22 = q_rot_11[(Ellipsis, slice(48, None, None))]
        q_rot_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_45 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_105 = cat_45 * sin_14
        cat_45 = None
        add_67 = mul_104 + mul_105
        mul_104 = mul_105 = None
        q_embed_11 = torch.cat([add_67, q_pass_11], dim=-1)
        add_67 = q_pass_11 = None
        mul_106 = k_rot_11 * cos_14
        cos_14 = None
        x1_23 = k_rot_11[(Ellipsis, slice(None, 48, None))]
        x2_23 = k_rot_11[(Ellipsis, slice(48, None, None))]
        k_rot_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_47 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_107 = cat_47 * sin_14
        cat_47 = sin_14 = None
        add_68 = mul_106 + mul_107
        mul_106 = mul_107 = None
        k_embed_11 = torch.cat([add_68, k_pass_11], dim=-1)
        add_68 = k_pass_11 = None
        attention_mask_12 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_11 = q_embed_11.contiguous()
        q_embed_11 = None
        key_11 = k_embed_11.contiguous()
        value_11 = value_states_23.contiguous()
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_11 = key_11 = value_11 = attention_mask_12 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_11 = attn_output_45.reshape(1, 2, -1)
        attn_output_45 = None
        attn_output_46 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_22 = torch.nn.functional.dropout(attn_output_47, 0.0, False, False)
        attn_output_47 = None
        hidden_states_102 = hidden_states_98 + dropout_22
        hidden_states_98 = dropout_22 = None
        hidden_states_103 = hidden_states_102.to(torch.float32)
        pow_24 = hidden_states_103.pow(2)
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_70 = variance_23 + 1e-05
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_104 = hidden_states_103 * rsqrt_23
        hidden_states_103 = rsqrt_23 = None
        to_51 = hidden_states_104.to(torch.bfloat16)
        hidden_states_104 = None
        hidden_states_105 = (
            l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_51
        ) = None
        up_states_33 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_layers_modules_11_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_105 = l_self_modules_layers_modules_11_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_11 = up_states_33.chunk(2, dim=-1)
        up_states_33 = None
        gate_11 = chunk_11[0]
        up_states_34 = chunk_11[1]
        chunk_11 = None
        silu_11 = torch.nn.functional.silu(gate_11, inplace=False)
        gate_11 = None
        up_states_35 = up_states_34 * silu_11
        up_states_34 = silu_11 = None
        hidden_states_106 = torch._C._nn.linear(
            up_states_35,
            l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_35 = l_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_23 = torch.nn.functional.dropout(hidden_states_106, 0.0, False, False)
        hidden_states_106 = None
        hidden_states_107 = hidden_states_102 + dropout_23
        hidden_states_102 = dropout_23 = None
        hidden_states_108 = hidden_states_107.to(torch.float32)
        pow_25 = hidden_states_108.pow(2)
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_72 = variance_24 + 1e-05
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_109 = hidden_states_108 * rsqrt_24
        hidden_states_108 = rsqrt_24 = None
        to_53 = hidden_states_109.to(torch.bfloat16)
        hidden_states_109 = None
        hidden_states_110 = (
            l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
            * to_53
        )
        l_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            to_53
        ) = None
        qkv_12 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_layers_modules_12_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_110 = l_self_modules_layers_modules_12_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_24 = qkv_12[(Ellipsis, slice(None, 3072, None))]
        key_states_24 = qkv_12[(Ellipsis, slice(3072, 6144, None))]
        value_states_24 = qkv_12[(Ellipsis, slice(6144, None, None))]
        qkv_12 = None
        view_37 = query_states_24.view((1, 2, -1, 96))
        query_states_24 = None
        query_states_25 = view_37.transpose(1, 2)
        view_37 = None
        view_38 = key_states_24.view((1, 2, -1, 96))
        key_states_24 = None
        key_states_25 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = value_states_24.view((1, 2, -1, 96))
        value_states_24 = None
        value_states_25 = view_39.transpose(1, 2)
        view_39 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        q_rot_12 = query_states_25[(Ellipsis, slice(None, 96, None))]
        q_pass_12 = query_states_25[(Ellipsis, slice(96, None, None))]
        query_states_25 = None
        k_rot_12 = key_states_25[(Ellipsis, slice(None, 96, None))]
        k_pass_12 = key_states_25[(Ellipsis, slice(96, None, None))]
        key_states_25 = None
        mul_113 = q_rot_12 * cos_15
        x1_24 = q_rot_12[(Ellipsis, slice(None, 48, None))]
        x2_24 = q_rot_12[(Ellipsis, slice(48, None, None))]
        q_rot_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_49 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_114 = cat_49 * sin_15
        cat_49 = None
        add_73 = mul_113 + mul_114
        mul_113 = mul_114 = None
        q_embed_12 = torch.cat([add_73, q_pass_12], dim=-1)
        add_73 = q_pass_12 = None
        mul_115 = k_rot_12 * cos_15
        cos_15 = None
        x1_25 = k_rot_12[(Ellipsis, slice(None, 48, None))]
        x2_25 = k_rot_12[(Ellipsis, slice(48, None, None))]
        k_rot_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_51 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_116 = cat_51 * sin_15
        cat_51 = sin_15 = None
        add_74 = mul_115 + mul_116
        mul_115 = mul_116 = None
        k_embed_12 = torch.cat([add_74, k_pass_12], dim=-1)
        add_74 = k_pass_12 = None
        attention_mask_13 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_12 = q_embed_12.contiguous()
        q_embed_12 = None
        key_12 = k_embed_12.contiguous()
        value_12 = value_states_25.contiguous()
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_12,
            value_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_12 = key_12 = value_12 = attention_mask_13 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_12 = attn_output_49.reshape(1, 2, -1)
        attn_output_49 = None
        attn_output_50 = reshape_12.contiguous()
        reshape_12 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_24 = torch.nn.functional.dropout(attn_output_51, 0.0, False, False)
        attn_output_51 = None
        hidden_states_111 = hidden_states_107 + dropout_24
        hidden_states_107 = dropout_24 = None
        hidden_states_112 = hidden_states_111.to(torch.float32)
        pow_26 = hidden_states_112.pow(2)
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_76 = variance_25 + 1e-05
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_76)
        add_76 = None
        hidden_states_113 = hidden_states_112 * rsqrt_25
        hidden_states_112 = rsqrt_25 = None
        to_55 = hidden_states_113.to(torch.bfloat16)
        hidden_states_113 = None
        hidden_states_114 = (
            l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_55
        )
        l_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_55
        ) = None
        up_states_36 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_layers_modules_12_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_114 = l_self_modules_layers_modules_12_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_12 = up_states_36.chunk(2, dim=-1)
        up_states_36 = None
        gate_12 = chunk_12[0]
        up_states_37 = chunk_12[1]
        chunk_12 = None
        silu_12 = torch.nn.functional.silu(gate_12, inplace=False)
        gate_12 = None
        up_states_38 = up_states_37 * silu_12
        up_states_37 = silu_12 = None
        hidden_states_115 = torch._C._nn.linear(
            up_states_38,
            l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_38 = l_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_25 = torch.nn.functional.dropout(hidden_states_115, 0.0, False, False)
        hidden_states_115 = None
        hidden_states_116 = hidden_states_111 + dropout_25
        hidden_states_111 = dropout_25 = None
        hidden_states_117 = hidden_states_116.to(torch.float32)
        pow_27 = hidden_states_117.pow(2)
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_78 = variance_26 + 1e-05
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_118 = hidden_states_117 * rsqrt_26
        hidden_states_117 = rsqrt_26 = None
        to_57 = hidden_states_118.to(torch.bfloat16)
        hidden_states_118 = None
        hidden_states_119 = (
            l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
            * to_57
        )
        l_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            to_57
        ) = None
        qkv_13 = torch._C._nn.linear(
            hidden_states_119,
            l_self_modules_layers_modules_13_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_119 = l_self_modules_layers_modules_13_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_26 = qkv_13[(Ellipsis, slice(None, 3072, None))]
        key_states_26 = qkv_13[(Ellipsis, slice(3072, 6144, None))]
        value_states_26 = qkv_13[(Ellipsis, slice(6144, None, None))]
        qkv_13 = None
        view_40 = query_states_26.view((1, 2, -1, 96))
        query_states_26 = None
        query_states_27 = view_40.transpose(1, 2)
        view_40 = None
        view_41 = key_states_26.view((1, 2, -1, 96))
        key_states_26 = None
        key_states_27 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = value_states_26.view((1, 2, -1, 96))
        value_states_26 = None
        value_states_27 = view_42.transpose(1, 2)
        view_42 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        q_rot_13 = query_states_27[(Ellipsis, slice(None, 96, None))]
        q_pass_13 = query_states_27[(Ellipsis, slice(96, None, None))]
        query_states_27 = None
        k_rot_13 = key_states_27[(Ellipsis, slice(None, 96, None))]
        k_pass_13 = key_states_27[(Ellipsis, slice(96, None, None))]
        key_states_27 = None
        mul_122 = q_rot_13 * cos_16
        x1_26 = q_rot_13[(Ellipsis, slice(None, 48, None))]
        x2_26 = q_rot_13[(Ellipsis, slice(48, None, None))]
        q_rot_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_53 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_123 = cat_53 * sin_16
        cat_53 = None
        add_79 = mul_122 + mul_123
        mul_122 = mul_123 = None
        q_embed_13 = torch.cat([add_79, q_pass_13], dim=-1)
        add_79 = q_pass_13 = None
        mul_124 = k_rot_13 * cos_16
        cos_16 = None
        x1_27 = k_rot_13[(Ellipsis, slice(None, 48, None))]
        x2_27 = k_rot_13[(Ellipsis, slice(48, None, None))]
        k_rot_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_55 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_125 = cat_55 * sin_16
        cat_55 = sin_16 = None
        add_80 = mul_124 + mul_125
        mul_124 = mul_125 = None
        k_embed_13 = torch.cat([add_80, k_pass_13], dim=-1)
        add_80 = k_pass_13 = None
        attention_mask_14 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_13 = q_embed_13.contiguous()
        q_embed_13 = None
        key_13 = k_embed_13.contiguous()
        value_13 = value_states_27.contiguous()
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_13,
            value_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_13 = key_13 = value_13 = attention_mask_14 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_13 = attn_output_53.reshape(1, 2, -1)
        attn_output_53 = None
        attn_output_54 = reshape_13.contiguous()
        reshape_13 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_26 = torch.nn.functional.dropout(attn_output_55, 0.0, False, False)
        attn_output_55 = None
        hidden_states_120 = hidden_states_116 + dropout_26
        hidden_states_116 = dropout_26 = None
        hidden_states_121 = hidden_states_120.to(torch.float32)
        pow_28 = hidden_states_121.pow(2)
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_82 = variance_27 + 1e-05
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_122 = hidden_states_121 * rsqrt_27
        hidden_states_121 = rsqrt_27 = None
        to_59 = hidden_states_122.to(torch.bfloat16)
        hidden_states_122 = None
        hidden_states_123 = (
            l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_59
        )
        l_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_59
        ) = None
        up_states_39 = torch._C._nn.linear(
            hidden_states_123,
            l_self_modules_layers_modules_13_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_123 = l_self_modules_layers_modules_13_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_13 = up_states_39.chunk(2, dim=-1)
        up_states_39 = None
        gate_13 = chunk_13[0]
        up_states_40 = chunk_13[1]
        chunk_13 = None
        silu_13 = torch.nn.functional.silu(gate_13, inplace=False)
        gate_13 = None
        up_states_41 = up_states_40 * silu_13
        up_states_40 = silu_13 = None
        hidden_states_124 = torch._C._nn.linear(
            up_states_41,
            l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_41 = l_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_27 = torch.nn.functional.dropout(hidden_states_124, 0.0, False, False)
        hidden_states_124 = None
        hidden_states_125 = hidden_states_120 + dropout_27
        hidden_states_120 = dropout_27 = None
        hidden_states_126 = hidden_states_125.to(torch.float32)
        pow_29 = hidden_states_126.pow(2)
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_84 = variance_28 + 1e-05
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_84)
        add_84 = None
        hidden_states_127 = hidden_states_126 * rsqrt_28
        hidden_states_126 = rsqrt_28 = None
        to_61 = hidden_states_127.to(torch.bfloat16)
        hidden_states_127 = None
        hidden_states_128 = (
            l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
            * to_61
        )
        l_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            to_61
        ) = None
        qkv_14 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_layers_modules_14_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_128 = l_self_modules_layers_modules_14_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_28 = qkv_14[(Ellipsis, slice(None, 3072, None))]
        key_states_28 = qkv_14[(Ellipsis, slice(3072, 6144, None))]
        value_states_28 = qkv_14[(Ellipsis, slice(6144, None, None))]
        qkv_14 = None
        view_43 = query_states_28.view((1, 2, -1, 96))
        query_states_28 = None
        query_states_29 = view_43.transpose(1, 2)
        view_43 = None
        view_44 = key_states_28.view((1, 2, -1, 96))
        key_states_28 = None
        key_states_29 = view_44.transpose(1, 2)
        view_44 = None
        view_45 = value_states_28.view((1, 2, -1, 96))
        value_states_28 = None
        value_states_29 = view_45.transpose(1, 2)
        view_45 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        q_rot_14 = query_states_29[(Ellipsis, slice(None, 96, None))]
        q_pass_14 = query_states_29[(Ellipsis, slice(96, None, None))]
        query_states_29 = None
        k_rot_14 = key_states_29[(Ellipsis, slice(None, 96, None))]
        k_pass_14 = key_states_29[(Ellipsis, slice(96, None, None))]
        key_states_29 = None
        mul_131 = q_rot_14 * cos_17
        x1_28 = q_rot_14[(Ellipsis, slice(None, 48, None))]
        x2_28 = q_rot_14[(Ellipsis, slice(48, None, None))]
        q_rot_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_57 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_132 = cat_57 * sin_17
        cat_57 = None
        add_85 = mul_131 + mul_132
        mul_131 = mul_132 = None
        q_embed_14 = torch.cat([add_85, q_pass_14], dim=-1)
        add_85 = q_pass_14 = None
        mul_133 = k_rot_14 * cos_17
        cos_17 = None
        x1_29 = k_rot_14[(Ellipsis, slice(None, 48, None))]
        x2_29 = k_rot_14[(Ellipsis, slice(48, None, None))]
        k_rot_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_59 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_134 = cat_59 * sin_17
        cat_59 = sin_17 = None
        add_86 = mul_133 + mul_134
        mul_133 = mul_134 = None
        k_embed_14 = torch.cat([add_86, k_pass_14], dim=-1)
        add_86 = k_pass_14 = None
        attention_mask_15 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_14 = q_embed_14.contiguous()
        q_embed_14 = None
        key_14 = k_embed_14.contiguous()
        value_14 = value_states_29.contiguous()
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_14,
            value_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_14 = key_14 = value_14 = attention_mask_15 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_14 = attn_output_57.reshape(1, 2, -1)
        attn_output_57 = None
        attn_output_58 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_28 = torch.nn.functional.dropout(attn_output_59, 0.0, False, False)
        attn_output_59 = None
        hidden_states_129 = hidden_states_125 + dropout_28
        hidden_states_125 = dropout_28 = None
        hidden_states_130 = hidden_states_129.to(torch.float32)
        pow_30 = hidden_states_130.pow(2)
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_88 = variance_29 + 1e-05
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_131 = hidden_states_130 * rsqrt_29
        hidden_states_130 = rsqrt_29 = None
        to_63 = hidden_states_131.to(torch.bfloat16)
        hidden_states_131 = None
        hidden_states_132 = (
            l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_63
        )
        l_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_63
        ) = None
        up_states_42 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_layers_modules_14_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_132 = l_self_modules_layers_modules_14_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_14 = up_states_42.chunk(2, dim=-1)
        up_states_42 = None
        gate_14 = chunk_14[0]
        up_states_43 = chunk_14[1]
        chunk_14 = None
        silu_14 = torch.nn.functional.silu(gate_14, inplace=False)
        gate_14 = None
        up_states_44 = up_states_43 * silu_14
        up_states_43 = silu_14 = None
        hidden_states_133 = torch._C._nn.linear(
            up_states_44,
            l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_44 = l_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_29 = torch.nn.functional.dropout(hidden_states_133, 0.0, False, False)
        hidden_states_133 = None
        hidden_states_134 = hidden_states_129 + dropout_29
        hidden_states_129 = dropout_29 = None
        hidden_states_135 = hidden_states_134.to(torch.float32)
        pow_31 = hidden_states_135.pow(2)
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_90 = variance_30 + 1e-05
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_90)
        add_90 = None
        hidden_states_136 = hidden_states_135 * rsqrt_30
        hidden_states_135 = rsqrt_30 = None
        to_65 = hidden_states_136.to(torch.bfloat16)
        hidden_states_136 = None
        hidden_states_137 = (
            l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
            * to_65
        )
        l_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            to_65
        ) = None
        qkv_15 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_layers_modules_15_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_137 = l_self_modules_layers_modules_15_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_30 = qkv_15[(Ellipsis, slice(None, 3072, None))]
        key_states_30 = qkv_15[(Ellipsis, slice(3072, 6144, None))]
        value_states_30 = qkv_15[(Ellipsis, slice(6144, None, None))]
        qkv_15 = None
        view_46 = query_states_30.view((1, 2, -1, 96))
        query_states_30 = None
        query_states_31 = view_46.transpose(1, 2)
        view_46 = None
        view_47 = key_states_30.view((1, 2, -1, 96))
        key_states_30 = None
        key_states_31 = view_47.transpose(1, 2)
        view_47 = None
        view_48 = value_states_30.view((1, 2, -1, 96))
        value_states_30 = None
        value_states_31 = view_48.transpose(1, 2)
        view_48 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        q_rot_15 = query_states_31[(Ellipsis, slice(None, 96, None))]
        q_pass_15 = query_states_31[(Ellipsis, slice(96, None, None))]
        query_states_31 = None
        k_rot_15 = key_states_31[(Ellipsis, slice(None, 96, None))]
        k_pass_15 = key_states_31[(Ellipsis, slice(96, None, None))]
        key_states_31 = None
        mul_140 = q_rot_15 * cos_18
        x1_30 = q_rot_15[(Ellipsis, slice(None, 48, None))]
        x2_30 = q_rot_15[(Ellipsis, slice(48, None, None))]
        q_rot_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_61 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_141 = cat_61 * sin_18
        cat_61 = None
        add_91 = mul_140 + mul_141
        mul_140 = mul_141 = None
        q_embed_15 = torch.cat([add_91, q_pass_15], dim=-1)
        add_91 = q_pass_15 = None
        mul_142 = k_rot_15 * cos_18
        cos_18 = None
        x1_31 = k_rot_15[(Ellipsis, slice(None, 48, None))]
        x2_31 = k_rot_15[(Ellipsis, slice(48, None, None))]
        k_rot_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_63 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_143 = cat_63 * sin_18
        cat_63 = sin_18 = None
        add_92 = mul_142 + mul_143
        mul_142 = mul_143 = None
        k_embed_15 = torch.cat([add_92, k_pass_15], dim=-1)
        add_92 = k_pass_15 = None
        attention_mask_16 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_15 = q_embed_15.contiguous()
        q_embed_15 = None
        key_15 = k_embed_15.contiguous()
        value_15 = value_states_31.contiguous()
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_15,
            value_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_15 = key_15 = value_15 = attention_mask_16 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_15 = attn_output_61.reshape(1, 2, -1)
        attn_output_61 = None
        attn_output_62 = reshape_15.contiguous()
        reshape_15 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_30 = torch.nn.functional.dropout(attn_output_63, 0.0, False, False)
        attn_output_63 = None
        hidden_states_138 = hidden_states_134 + dropout_30
        hidden_states_134 = dropout_30 = None
        hidden_states_139 = hidden_states_138.to(torch.float32)
        pow_32 = hidden_states_139.pow(2)
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_94 = variance_31 + 1e-05
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_140 = hidden_states_139 * rsqrt_31
        hidden_states_139 = rsqrt_31 = None
        to_67 = hidden_states_140.to(torch.bfloat16)
        hidden_states_140 = None
        hidden_states_141 = (
            l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_67
        )
        l_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_67
        ) = None
        up_states_45 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_layers_modules_15_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_141 = l_self_modules_layers_modules_15_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_15 = up_states_45.chunk(2, dim=-1)
        up_states_45 = None
        gate_15 = chunk_15[0]
        up_states_46 = chunk_15[1]
        chunk_15 = None
        silu_15 = torch.nn.functional.silu(gate_15, inplace=False)
        gate_15 = None
        up_states_47 = up_states_46 * silu_15
        up_states_46 = silu_15 = None
        hidden_states_142 = torch._C._nn.linear(
            up_states_47,
            l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_47 = l_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_31 = torch.nn.functional.dropout(hidden_states_142, 0.0, False, False)
        hidden_states_142 = None
        hidden_states_143 = hidden_states_138 + dropout_31
        hidden_states_138 = dropout_31 = None
        hidden_states_144 = hidden_states_143.to(torch.float32)
        pow_33 = hidden_states_144.pow(2)
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_96 = variance_32 + 1e-05
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_145 = hidden_states_144 * rsqrt_32
        hidden_states_144 = rsqrt_32 = None
        to_69 = hidden_states_145.to(torch.bfloat16)
        hidden_states_145 = None
        hidden_states_146 = (
            l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
            * to_69
        )
        l_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            to_69
        ) = None
        qkv_16 = torch._C._nn.linear(
            hidden_states_146,
            l_self_modules_layers_modules_16_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_146 = l_self_modules_layers_modules_16_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_32 = qkv_16[(Ellipsis, slice(None, 3072, None))]
        key_states_32 = qkv_16[(Ellipsis, slice(3072, 6144, None))]
        value_states_32 = qkv_16[(Ellipsis, slice(6144, None, None))]
        qkv_16 = None
        view_49 = query_states_32.view((1, 2, -1, 96))
        query_states_32 = None
        query_states_33 = view_49.transpose(1, 2)
        view_49 = None
        view_50 = key_states_32.view((1, 2, -1, 96))
        key_states_32 = None
        key_states_33 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = value_states_32.view((1, 2, -1, 96))
        value_states_32 = None
        value_states_33 = view_51.transpose(1, 2)
        view_51 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        q_rot_16 = query_states_33[(Ellipsis, slice(None, 96, None))]
        q_pass_16 = query_states_33[(Ellipsis, slice(96, None, None))]
        query_states_33 = None
        k_rot_16 = key_states_33[(Ellipsis, slice(None, 96, None))]
        k_pass_16 = key_states_33[(Ellipsis, slice(96, None, None))]
        key_states_33 = None
        mul_149 = q_rot_16 * cos_19
        x1_32 = q_rot_16[(Ellipsis, slice(None, 48, None))]
        x2_32 = q_rot_16[(Ellipsis, slice(48, None, None))]
        q_rot_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_65 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_150 = cat_65 * sin_19
        cat_65 = None
        add_97 = mul_149 + mul_150
        mul_149 = mul_150 = None
        q_embed_16 = torch.cat([add_97, q_pass_16], dim=-1)
        add_97 = q_pass_16 = None
        mul_151 = k_rot_16 * cos_19
        cos_19 = None
        x1_33 = k_rot_16[(Ellipsis, slice(None, 48, None))]
        x2_33 = k_rot_16[(Ellipsis, slice(48, None, None))]
        k_rot_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_67 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_152 = cat_67 * sin_19
        cat_67 = sin_19 = None
        add_98 = mul_151 + mul_152
        mul_151 = mul_152 = None
        k_embed_16 = torch.cat([add_98, k_pass_16], dim=-1)
        add_98 = k_pass_16 = None
        attention_mask_17 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_16 = q_embed_16.contiguous()
        q_embed_16 = None
        key_16 = k_embed_16.contiguous()
        value_16 = value_states_33.contiguous()
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_16,
            value_16,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_16 = key_16 = value_16 = attention_mask_17 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_16 = attn_output_65.reshape(1, 2, -1)
        attn_output_65 = None
        attn_output_66 = reshape_16.contiguous()
        reshape_16 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_32 = torch.nn.functional.dropout(attn_output_67, 0.0, False, False)
        attn_output_67 = None
        hidden_states_147 = hidden_states_143 + dropout_32
        hidden_states_143 = dropout_32 = None
        hidden_states_148 = hidden_states_147.to(torch.float32)
        pow_34 = hidden_states_148.pow(2)
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_100 = variance_33 + 1e-05
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_100)
        add_100 = None
        hidden_states_149 = hidden_states_148 * rsqrt_33
        hidden_states_148 = rsqrt_33 = None
        to_71 = hidden_states_149.to(torch.bfloat16)
        hidden_states_149 = None
        hidden_states_150 = (
            l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_71
        )
        l_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_71
        ) = None
        up_states_48 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_layers_modules_16_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_150 = l_self_modules_layers_modules_16_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_16 = up_states_48.chunk(2, dim=-1)
        up_states_48 = None
        gate_16 = chunk_16[0]
        up_states_49 = chunk_16[1]
        chunk_16 = None
        silu_16 = torch.nn.functional.silu(gate_16, inplace=False)
        gate_16 = None
        up_states_50 = up_states_49 * silu_16
        up_states_49 = silu_16 = None
        hidden_states_151 = torch._C._nn.linear(
            up_states_50,
            l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_50 = l_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_33 = torch.nn.functional.dropout(hidden_states_151, 0.0, False, False)
        hidden_states_151 = None
        hidden_states_152 = hidden_states_147 + dropout_33
        hidden_states_147 = dropout_33 = None
        hidden_states_153 = hidden_states_152.to(torch.float32)
        pow_35 = hidden_states_153.pow(2)
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_102 = variance_34 + 1e-05
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_154 = hidden_states_153 * rsqrt_34
        hidden_states_153 = rsqrt_34 = None
        to_73 = hidden_states_154.to(torch.bfloat16)
        hidden_states_154 = None
        hidden_states_155 = (
            l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
            * to_73
        )
        l_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            to_73
        ) = None
        qkv_17 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_layers_modules_17_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_155 = l_self_modules_layers_modules_17_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_34 = qkv_17[(Ellipsis, slice(None, 3072, None))]
        key_states_34 = qkv_17[(Ellipsis, slice(3072, 6144, None))]
        value_states_34 = qkv_17[(Ellipsis, slice(6144, None, None))]
        qkv_17 = None
        view_52 = query_states_34.view((1, 2, -1, 96))
        query_states_34 = None
        query_states_35 = view_52.transpose(1, 2)
        view_52 = None
        view_53 = key_states_34.view((1, 2, -1, 96))
        key_states_34 = None
        key_states_35 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = value_states_34.view((1, 2, -1, 96))
        value_states_34 = None
        value_states_35 = view_54.transpose(1, 2)
        view_54 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        q_rot_17 = query_states_35[(Ellipsis, slice(None, 96, None))]
        q_pass_17 = query_states_35[(Ellipsis, slice(96, None, None))]
        query_states_35 = None
        k_rot_17 = key_states_35[(Ellipsis, slice(None, 96, None))]
        k_pass_17 = key_states_35[(Ellipsis, slice(96, None, None))]
        key_states_35 = None
        mul_158 = q_rot_17 * cos_20
        x1_34 = q_rot_17[(Ellipsis, slice(None, 48, None))]
        x2_34 = q_rot_17[(Ellipsis, slice(48, None, None))]
        q_rot_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_69 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_159 = cat_69 * sin_20
        cat_69 = None
        add_103 = mul_158 + mul_159
        mul_158 = mul_159 = None
        q_embed_17 = torch.cat([add_103, q_pass_17], dim=-1)
        add_103 = q_pass_17 = None
        mul_160 = k_rot_17 * cos_20
        cos_20 = None
        x1_35 = k_rot_17[(Ellipsis, slice(None, 48, None))]
        x2_35 = k_rot_17[(Ellipsis, slice(48, None, None))]
        k_rot_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_71 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_161 = cat_71 * sin_20
        cat_71 = sin_20 = None
        add_104 = mul_160 + mul_161
        mul_160 = mul_161 = None
        k_embed_17 = torch.cat([add_104, k_pass_17], dim=-1)
        add_104 = k_pass_17 = None
        attention_mask_18 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_17 = q_embed_17.contiguous()
        q_embed_17 = None
        key_17 = k_embed_17.contiguous()
        value_17 = value_states_35.contiguous()
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_17,
            value_17,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_17 = key_17 = value_17 = attention_mask_18 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_17 = attn_output_69.reshape(1, 2, -1)
        attn_output_69 = None
        attn_output_70 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_34 = torch.nn.functional.dropout(attn_output_71, 0.0, False, False)
        attn_output_71 = None
        hidden_states_156 = hidden_states_152 + dropout_34
        hidden_states_152 = dropout_34 = None
        hidden_states_157 = hidden_states_156.to(torch.float32)
        pow_36 = hidden_states_157.pow(2)
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_106 = variance_35 + 1e-05
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_158 = hidden_states_157 * rsqrt_35
        hidden_states_157 = rsqrt_35 = None
        to_75 = hidden_states_158.to(torch.bfloat16)
        hidden_states_158 = None
        hidden_states_159 = (
            l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_75
        )
        l_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_75
        ) = None
        up_states_51 = torch._C._nn.linear(
            hidden_states_159,
            l_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_159 = l_self_modules_layers_modules_17_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_17 = up_states_51.chunk(2, dim=-1)
        up_states_51 = None
        gate_17 = chunk_17[0]
        up_states_52 = chunk_17[1]
        chunk_17 = None
        silu_17 = torch.nn.functional.silu(gate_17, inplace=False)
        gate_17 = None
        up_states_53 = up_states_52 * silu_17
        up_states_52 = silu_17 = None
        hidden_states_160 = torch._C._nn.linear(
            up_states_53,
            l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_53 = l_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_35 = torch.nn.functional.dropout(hidden_states_160, 0.0, False, False)
        hidden_states_160 = None
        hidden_states_161 = hidden_states_156 + dropout_35
        hidden_states_156 = dropout_35 = None
        hidden_states_162 = hidden_states_161.to(torch.float32)
        pow_37 = hidden_states_162.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_108 = variance_36 + 1e-05
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_163 = hidden_states_162 * rsqrt_36
        hidden_states_162 = rsqrt_36 = None
        to_77 = hidden_states_163.to(torch.bfloat16)
        hidden_states_163 = None
        hidden_states_164 = (
            l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
            * to_77
        )
        l_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            to_77
        ) = None
        qkv_18 = torch._C._nn.linear(
            hidden_states_164,
            l_self_modules_layers_modules_18_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_164 = l_self_modules_layers_modules_18_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_36 = qkv_18[(Ellipsis, slice(None, 3072, None))]
        key_states_36 = qkv_18[(Ellipsis, slice(3072, 6144, None))]
        value_states_36 = qkv_18[(Ellipsis, slice(6144, None, None))]
        qkv_18 = None
        view_55 = query_states_36.view((1, 2, -1, 96))
        query_states_36 = None
        query_states_37 = view_55.transpose(1, 2)
        view_55 = None
        view_56 = key_states_36.view((1, 2, -1, 96))
        key_states_36 = None
        key_states_37 = view_56.transpose(1, 2)
        view_56 = None
        view_57 = value_states_36.view((1, 2, -1, 96))
        value_states_36 = None
        value_states_37 = view_57.transpose(1, 2)
        view_57 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        q_rot_18 = query_states_37[(Ellipsis, slice(None, 96, None))]
        q_pass_18 = query_states_37[(Ellipsis, slice(96, None, None))]
        query_states_37 = None
        k_rot_18 = key_states_37[(Ellipsis, slice(None, 96, None))]
        k_pass_18 = key_states_37[(Ellipsis, slice(96, None, None))]
        key_states_37 = None
        mul_167 = q_rot_18 * cos_21
        x1_36 = q_rot_18[(Ellipsis, slice(None, 48, None))]
        x2_36 = q_rot_18[(Ellipsis, slice(48, None, None))]
        q_rot_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_73 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_168 = cat_73 * sin_21
        cat_73 = None
        add_109 = mul_167 + mul_168
        mul_167 = mul_168 = None
        q_embed_18 = torch.cat([add_109, q_pass_18], dim=-1)
        add_109 = q_pass_18 = None
        mul_169 = k_rot_18 * cos_21
        cos_21 = None
        x1_37 = k_rot_18[(Ellipsis, slice(None, 48, None))]
        x2_37 = k_rot_18[(Ellipsis, slice(48, None, None))]
        k_rot_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_75 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_170 = cat_75 * sin_21
        cat_75 = sin_21 = None
        add_110 = mul_169 + mul_170
        mul_169 = mul_170 = None
        k_embed_18 = torch.cat([add_110, k_pass_18], dim=-1)
        add_110 = k_pass_18 = None
        attention_mask_19 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_18 = q_embed_18.contiguous()
        q_embed_18 = None
        key_18 = k_embed_18.contiguous()
        value_18 = value_states_37.contiguous()
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_18,
            value_18,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_18 = key_18 = value_18 = attention_mask_19 = None
        transpose_76 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_76.contiguous()
        transpose_76 = None
        reshape_18 = attn_output_73.reshape(1, 2, -1)
        attn_output_73 = None
        attn_output_74 = reshape_18.contiguous()
        reshape_18 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_36 = torch.nn.functional.dropout(attn_output_75, 0.0, False, False)
        attn_output_75 = None
        hidden_states_165 = hidden_states_161 + dropout_36
        hidden_states_161 = dropout_36 = None
        hidden_states_166 = hidden_states_165.to(torch.float32)
        pow_38 = hidden_states_166.pow(2)
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_112 = variance_37 + 1e-05
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_167 = hidden_states_166 * rsqrt_37
        hidden_states_166 = rsqrt_37 = None
        to_79 = hidden_states_167.to(torch.bfloat16)
        hidden_states_167 = None
        hidden_states_168 = (
            l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
            * to_79
        )
        l_self_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            to_79
        ) = None
        up_states_54 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_layers_modules_18_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_168 = l_self_modules_layers_modules_18_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_18 = up_states_54.chunk(2, dim=-1)
        up_states_54 = None
        gate_18 = chunk_18[0]
        up_states_55 = chunk_18[1]
        chunk_18 = None
        silu_18 = torch.nn.functional.silu(gate_18, inplace=False)
        gate_18 = None
        up_states_56 = up_states_55 * silu_18
        up_states_55 = silu_18 = None
        hidden_states_169 = torch._C._nn.linear(
            up_states_56,
            l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_56 = l_self_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_37 = torch.nn.functional.dropout(hidden_states_169, 0.0, False, False)
        hidden_states_169 = None
        hidden_states_170 = hidden_states_165 + dropout_37
        hidden_states_165 = dropout_37 = None
        hidden_states_171 = hidden_states_170.to(torch.float32)
        pow_39 = hidden_states_171.pow(2)
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_114 = variance_38 + 1e-05
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_114)
        add_114 = None
        hidden_states_172 = hidden_states_171 * rsqrt_38
        hidden_states_171 = rsqrt_38 = None
        to_81 = hidden_states_172.to(torch.bfloat16)
        hidden_states_172 = None
        hidden_states_173 = (
            l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
            * to_81
        )
        l_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            to_81
        ) = None
        qkv_19 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_layers_modules_19_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_173 = l_self_modules_layers_modules_19_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_38 = qkv_19[(Ellipsis, slice(None, 3072, None))]
        key_states_38 = qkv_19[(Ellipsis, slice(3072, 6144, None))]
        value_states_38 = qkv_19[(Ellipsis, slice(6144, None, None))]
        qkv_19 = None
        view_58 = query_states_38.view((1, 2, -1, 96))
        query_states_38 = None
        query_states_39 = view_58.transpose(1, 2)
        view_58 = None
        view_59 = key_states_38.view((1, 2, -1, 96))
        key_states_38 = None
        key_states_39 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = value_states_38.view((1, 2, -1, 96))
        value_states_38 = None
        value_states_39 = view_60.transpose(1, 2)
        view_60 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        q_rot_19 = query_states_39[(Ellipsis, slice(None, 96, None))]
        q_pass_19 = query_states_39[(Ellipsis, slice(96, None, None))]
        query_states_39 = None
        k_rot_19 = key_states_39[(Ellipsis, slice(None, 96, None))]
        k_pass_19 = key_states_39[(Ellipsis, slice(96, None, None))]
        key_states_39 = None
        mul_176 = q_rot_19 * cos_22
        x1_38 = q_rot_19[(Ellipsis, slice(None, 48, None))]
        x2_38 = q_rot_19[(Ellipsis, slice(48, None, None))]
        q_rot_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_77 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_177 = cat_77 * sin_22
        cat_77 = None
        add_115 = mul_176 + mul_177
        mul_176 = mul_177 = None
        q_embed_19 = torch.cat([add_115, q_pass_19], dim=-1)
        add_115 = q_pass_19 = None
        mul_178 = k_rot_19 * cos_22
        cos_22 = None
        x1_39 = k_rot_19[(Ellipsis, slice(None, 48, None))]
        x2_39 = k_rot_19[(Ellipsis, slice(48, None, None))]
        k_rot_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_79 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_179 = cat_79 * sin_22
        cat_79 = sin_22 = None
        add_116 = mul_178 + mul_179
        mul_178 = mul_179 = None
        k_embed_19 = torch.cat([add_116, k_pass_19], dim=-1)
        add_116 = k_pass_19 = None
        attention_mask_20 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_19 = q_embed_19.contiguous()
        q_embed_19 = None
        key_19 = k_embed_19.contiguous()
        value_19 = value_states_39.contiguous()
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_19,
            value_19,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_19 = key_19 = value_19 = attention_mask_20 = None
        transpose_80 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_80.contiguous()
        transpose_80 = None
        reshape_19 = attn_output_77.reshape(1, 2, -1)
        attn_output_77 = None
        attn_output_78 = reshape_19.contiguous()
        reshape_19 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_38 = torch.nn.functional.dropout(attn_output_79, 0.0, False, False)
        attn_output_79 = None
        hidden_states_174 = hidden_states_170 + dropout_38
        hidden_states_170 = dropout_38 = None
        hidden_states_175 = hidden_states_174.to(torch.float32)
        pow_40 = hidden_states_175.pow(2)
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_118 = variance_39 + 1e-05
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_176 = hidden_states_175 * rsqrt_39
        hidden_states_175 = rsqrt_39 = None
        to_83 = hidden_states_176.to(torch.bfloat16)
        hidden_states_176 = None
        hidden_states_177 = (
            l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
            * to_83
        )
        l_self_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            to_83
        ) = None
        up_states_57 = torch._C._nn.linear(
            hidden_states_177,
            l_self_modules_layers_modules_19_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_177 = l_self_modules_layers_modules_19_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_19 = up_states_57.chunk(2, dim=-1)
        up_states_57 = None
        gate_19 = chunk_19[0]
        up_states_58 = chunk_19[1]
        chunk_19 = None
        silu_19 = torch.nn.functional.silu(gate_19, inplace=False)
        gate_19 = None
        up_states_59 = up_states_58 * silu_19
        up_states_58 = silu_19 = None
        hidden_states_178 = torch._C._nn.linear(
            up_states_59,
            l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_59 = l_self_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_39 = torch.nn.functional.dropout(hidden_states_178, 0.0, False, False)
        hidden_states_178 = None
        hidden_states_179 = hidden_states_174 + dropout_39
        hidden_states_174 = dropout_39 = None
        hidden_states_180 = hidden_states_179.to(torch.float32)
        pow_41 = hidden_states_180.pow(2)
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_120 = variance_40 + 1e-05
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_181 = hidden_states_180 * rsqrt_40
        hidden_states_180 = rsqrt_40 = None
        to_85 = hidden_states_181.to(torch.bfloat16)
        hidden_states_181 = None
        hidden_states_182 = (
            l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
            * to_85
        )
        l_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            to_85
        ) = None
        qkv_20 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_layers_modules_20_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_182 = l_self_modules_layers_modules_20_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_40 = qkv_20[(Ellipsis, slice(None, 3072, None))]
        key_states_40 = qkv_20[(Ellipsis, slice(3072, 6144, None))]
        value_states_40 = qkv_20[(Ellipsis, slice(6144, None, None))]
        qkv_20 = None
        view_61 = query_states_40.view((1, 2, -1, 96))
        query_states_40 = None
        query_states_41 = view_61.transpose(1, 2)
        view_61 = None
        view_62 = key_states_40.view((1, 2, -1, 96))
        key_states_40 = None
        key_states_41 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = value_states_40.view((1, 2, -1, 96))
        value_states_40 = None
        value_states_41 = view_63.transpose(1, 2)
        view_63 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        q_rot_20 = query_states_41[(Ellipsis, slice(None, 96, None))]
        q_pass_20 = query_states_41[(Ellipsis, slice(96, None, None))]
        query_states_41 = None
        k_rot_20 = key_states_41[(Ellipsis, slice(None, 96, None))]
        k_pass_20 = key_states_41[(Ellipsis, slice(96, None, None))]
        key_states_41 = None
        mul_185 = q_rot_20 * cos_23
        x1_40 = q_rot_20[(Ellipsis, slice(None, 48, None))]
        x2_40 = q_rot_20[(Ellipsis, slice(48, None, None))]
        q_rot_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_81 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_186 = cat_81 * sin_23
        cat_81 = None
        add_121 = mul_185 + mul_186
        mul_185 = mul_186 = None
        q_embed_20 = torch.cat([add_121, q_pass_20], dim=-1)
        add_121 = q_pass_20 = None
        mul_187 = k_rot_20 * cos_23
        cos_23 = None
        x1_41 = k_rot_20[(Ellipsis, slice(None, 48, None))]
        x2_41 = k_rot_20[(Ellipsis, slice(48, None, None))]
        k_rot_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_83 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_188 = cat_83 * sin_23
        cat_83 = sin_23 = None
        add_122 = mul_187 + mul_188
        mul_187 = mul_188 = None
        k_embed_20 = torch.cat([add_122, k_pass_20], dim=-1)
        add_122 = k_pass_20 = None
        attention_mask_21 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_20 = q_embed_20.contiguous()
        q_embed_20 = None
        key_20 = k_embed_20.contiguous()
        value_20 = value_states_41.contiguous()
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_20,
            value_20,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_20 = key_20 = value_20 = attention_mask_21 = None
        transpose_84 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_84.contiguous()
        transpose_84 = None
        reshape_20 = attn_output_81.reshape(1, 2, -1)
        attn_output_81 = None
        attn_output_82 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_40 = torch.nn.functional.dropout(attn_output_83, 0.0, False, False)
        attn_output_83 = None
        hidden_states_183 = hidden_states_179 + dropout_40
        hidden_states_179 = dropout_40 = None
        hidden_states_184 = hidden_states_183.to(torch.float32)
        pow_42 = hidden_states_184.pow(2)
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_124 = variance_41 + 1e-05
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_185 = hidden_states_184 * rsqrt_41
        hidden_states_184 = rsqrt_41 = None
        to_87 = hidden_states_185.to(torch.bfloat16)
        hidden_states_185 = None
        hidden_states_186 = (
            l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
            * to_87
        )
        l_self_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            to_87
        ) = None
        up_states_60 = torch._C._nn.linear(
            hidden_states_186,
            l_self_modules_layers_modules_20_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_186 = l_self_modules_layers_modules_20_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_20 = up_states_60.chunk(2, dim=-1)
        up_states_60 = None
        gate_20 = chunk_20[0]
        up_states_61 = chunk_20[1]
        chunk_20 = None
        silu_20 = torch.nn.functional.silu(gate_20, inplace=False)
        gate_20 = None
        up_states_62 = up_states_61 * silu_20
        up_states_61 = silu_20 = None
        hidden_states_187 = torch._C._nn.linear(
            up_states_62,
            l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_62 = l_self_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_41 = torch.nn.functional.dropout(hidden_states_187, 0.0, False, False)
        hidden_states_187 = None
        hidden_states_188 = hidden_states_183 + dropout_41
        hidden_states_183 = dropout_41 = None
        hidden_states_189 = hidden_states_188.to(torch.float32)
        pow_43 = hidden_states_189.pow(2)
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_126 = variance_42 + 1e-05
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_190 = hidden_states_189 * rsqrt_42
        hidden_states_189 = rsqrt_42 = None
        to_89 = hidden_states_190.to(torch.bfloat16)
        hidden_states_190 = None
        hidden_states_191 = (
            l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
            * to_89
        )
        l_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            to_89
        ) = None
        qkv_21 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_layers_modules_21_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_191 = l_self_modules_layers_modules_21_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_42 = qkv_21[(Ellipsis, slice(None, 3072, None))]
        key_states_42 = qkv_21[(Ellipsis, slice(3072, 6144, None))]
        value_states_42 = qkv_21[(Ellipsis, slice(6144, None, None))]
        qkv_21 = None
        view_64 = query_states_42.view((1, 2, -1, 96))
        query_states_42 = None
        query_states_43 = view_64.transpose(1, 2)
        view_64 = None
        view_65 = key_states_42.view((1, 2, -1, 96))
        key_states_42 = None
        key_states_43 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = value_states_42.view((1, 2, -1, 96))
        value_states_42 = None
        value_states_43 = view_66.transpose(1, 2)
        view_66 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        q_rot_21 = query_states_43[(Ellipsis, slice(None, 96, None))]
        q_pass_21 = query_states_43[(Ellipsis, slice(96, None, None))]
        query_states_43 = None
        k_rot_21 = key_states_43[(Ellipsis, slice(None, 96, None))]
        k_pass_21 = key_states_43[(Ellipsis, slice(96, None, None))]
        key_states_43 = None
        mul_194 = q_rot_21 * cos_24
        x1_42 = q_rot_21[(Ellipsis, slice(None, 48, None))]
        x2_42 = q_rot_21[(Ellipsis, slice(48, None, None))]
        q_rot_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_85 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_195 = cat_85 * sin_24
        cat_85 = None
        add_127 = mul_194 + mul_195
        mul_194 = mul_195 = None
        q_embed_21 = torch.cat([add_127, q_pass_21], dim=-1)
        add_127 = q_pass_21 = None
        mul_196 = k_rot_21 * cos_24
        cos_24 = None
        x1_43 = k_rot_21[(Ellipsis, slice(None, 48, None))]
        x2_43 = k_rot_21[(Ellipsis, slice(48, None, None))]
        k_rot_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_87 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_197 = cat_87 * sin_24
        cat_87 = sin_24 = None
        add_128 = mul_196 + mul_197
        mul_196 = mul_197 = None
        k_embed_21 = torch.cat([add_128, k_pass_21], dim=-1)
        add_128 = k_pass_21 = None
        attention_mask_22 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_21 = q_embed_21.contiguous()
        q_embed_21 = None
        key_21 = k_embed_21.contiguous()
        value_21 = value_states_43.contiguous()
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_21,
            value_21,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_21 = key_21 = value_21 = attention_mask_22 = None
        transpose_88 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_88.contiguous()
        transpose_88 = None
        reshape_21 = attn_output_85.reshape(1, 2, -1)
        attn_output_85 = None
        attn_output_86 = reshape_21.contiguous()
        reshape_21 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_42 = torch.nn.functional.dropout(attn_output_87, 0.0, False, False)
        attn_output_87 = None
        hidden_states_192 = hidden_states_188 + dropout_42
        hidden_states_188 = dropout_42 = None
        hidden_states_193 = hidden_states_192.to(torch.float32)
        pow_44 = hidden_states_193.pow(2)
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_130 = variance_43 + 1e-05
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_130)
        add_130 = None
        hidden_states_194 = hidden_states_193 * rsqrt_43
        hidden_states_193 = rsqrt_43 = None
        to_91 = hidden_states_194.to(torch.bfloat16)
        hidden_states_194 = None
        hidden_states_195 = (
            l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
            * to_91
        )
        l_self_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            to_91
        ) = None
        up_states_63 = torch._C._nn.linear(
            hidden_states_195,
            l_self_modules_layers_modules_21_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_195 = l_self_modules_layers_modules_21_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_21 = up_states_63.chunk(2, dim=-1)
        up_states_63 = None
        gate_21 = chunk_21[0]
        up_states_64 = chunk_21[1]
        chunk_21 = None
        silu_21 = torch.nn.functional.silu(gate_21, inplace=False)
        gate_21 = None
        up_states_65 = up_states_64 * silu_21
        up_states_64 = silu_21 = None
        hidden_states_196 = torch._C._nn.linear(
            up_states_65,
            l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_65 = l_self_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_43 = torch.nn.functional.dropout(hidden_states_196, 0.0, False, False)
        hidden_states_196 = None
        hidden_states_197 = hidden_states_192 + dropout_43
        hidden_states_192 = dropout_43 = None
        hidden_states_198 = hidden_states_197.to(torch.float32)
        pow_45 = hidden_states_198.pow(2)
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_132 = variance_44 + 1e-05
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_132)
        add_132 = None
        hidden_states_199 = hidden_states_198 * rsqrt_44
        hidden_states_198 = rsqrt_44 = None
        to_93 = hidden_states_199.to(torch.bfloat16)
        hidden_states_199 = None
        hidden_states_200 = (
            l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
            * to_93
        )
        l_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            to_93
        ) = None
        qkv_22 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_layers_modules_22_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_200 = l_self_modules_layers_modules_22_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_44 = qkv_22[(Ellipsis, slice(None, 3072, None))]
        key_states_44 = qkv_22[(Ellipsis, slice(3072, 6144, None))]
        value_states_44 = qkv_22[(Ellipsis, slice(6144, None, None))]
        qkv_22 = None
        view_67 = query_states_44.view((1, 2, -1, 96))
        query_states_44 = None
        query_states_45 = view_67.transpose(1, 2)
        view_67 = None
        view_68 = key_states_44.view((1, 2, -1, 96))
        key_states_44 = None
        key_states_45 = view_68.transpose(1, 2)
        view_68 = None
        view_69 = value_states_44.view((1, 2, -1, 96))
        value_states_44 = None
        value_states_45 = view_69.transpose(1, 2)
        view_69 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        q_rot_22 = query_states_45[(Ellipsis, slice(None, 96, None))]
        q_pass_22 = query_states_45[(Ellipsis, slice(96, None, None))]
        query_states_45 = None
        k_rot_22 = key_states_45[(Ellipsis, slice(None, 96, None))]
        k_pass_22 = key_states_45[(Ellipsis, slice(96, None, None))]
        key_states_45 = None
        mul_203 = q_rot_22 * cos_25
        x1_44 = q_rot_22[(Ellipsis, slice(None, 48, None))]
        x2_44 = q_rot_22[(Ellipsis, slice(48, None, None))]
        q_rot_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_89 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_204 = cat_89 * sin_25
        cat_89 = None
        add_133 = mul_203 + mul_204
        mul_203 = mul_204 = None
        q_embed_22 = torch.cat([add_133, q_pass_22], dim=-1)
        add_133 = q_pass_22 = None
        mul_205 = k_rot_22 * cos_25
        cos_25 = None
        x1_45 = k_rot_22[(Ellipsis, slice(None, 48, None))]
        x2_45 = k_rot_22[(Ellipsis, slice(48, None, None))]
        k_rot_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_91 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_206 = cat_91 * sin_25
        cat_91 = sin_25 = None
        add_134 = mul_205 + mul_206
        mul_205 = mul_206 = None
        k_embed_22 = torch.cat([add_134, k_pass_22], dim=-1)
        add_134 = k_pass_22 = None
        attention_mask_23 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_22 = q_embed_22.contiguous()
        q_embed_22 = None
        key_22 = k_embed_22.contiguous()
        value_22 = value_states_45.contiguous()
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_22,
            value_22,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_22 = key_22 = value_22 = attention_mask_23 = None
        transpose_92 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_92.contiguous()
        transpose_92 = None
        reshape_22 = attn_output_89.reshape(1, 2, -1)
        attn_output_89 = None
        attn_output_90 = reshape_22.contiguous()
        reshape_22 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_44 = torch.nn.functional.dropout(attn_output_91, 0.0, False, False)
        attn_output_91 = None
        hidden_states_201 = hidden_states_197 + dropout_44
        hidden_states_197 = dropout_44 = None
        hidden_states_202 = hidden_states_201.to(torch.float32)
        pow_46 = hidden_states_202.pow(2)
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_136 = variance_45 + 1e-05
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_203 = hidden_states_202 * rsqrt_45
        hidden_states_202 = rsqrt_45 = None
        to_95 = hidden_states_203.to(torch.bfloat16)
        hidden_states_203 = None
        hidden_states_204 = (
            l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
            * to_95
        )
        l_self_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            to_95
        ) = None
        up_states_66 = torch._C._nn.linear(
            hidden_states_204,
            l_self_modules_layers_modules_22_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_204 = l_self_modules_layers_modules_22_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_22 = up_states_66.chunk(2, dim=-1)
        up_states_66 = None
        gate_22 = chunk_22[0]
        up_states_67 = chunk_22[1]
        chunk_22 = None
        silu_22 = torch.nn.functional.silu(gate_22, inplace=False)
        gate_22 = None
        up_states_68 = up_states_67 * silu_22
        up_states_67 = silu_22 = None
        hidden_states_205 = torch._C._nn.linear(
            up_states_68,
            l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_68 = l_self_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_45 = torch.nn.functional.dropout(hidden_states_205, 0.0, False, False)
        hidden_states_205 = None
        hidden_states_206 = hidden_states_201 + dropout_45
        hidden_states_201 = dropout_45 = None
        hidden_states_207 = hidden_states_206.to(torch.float32)
        pow_47 = hidden_states_207.pow(2)
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_138 = variance_46 + 1e-05
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_208 = hidden_states_207 * rsqrt_46
        hidden_states_207 = rsqrt_46 = None
        to_97 = hidden_states_208.to(torch.bfloat16)
        hidden_states_208 = None
        hidden_states_209 = (
            l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
            * to_97
        )
        l_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            to_97
        ) = None
        qkv_23 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_layers_modules_23_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_209 = l_self_modules_layers_modules_23_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_46 = qkv_23[(Ellipsis, slice(None, 3072, None))]
        key_states_46 = qkv_23[(Ellipsis, slice(3072, 6144, None))]
        value_states_46 = qkv_23[(Ellipsis, slice(6144, None, None))]
        qkv_23 = None
        view_70 = query_states_46.view((1, 2, -1, 96))
        query_states_46 = None
        query_states_47 = view_70.transpose(1, 2)
        view_70 = None
        view_71 = key_states_46.view((1, 2, -1, 96))
        key_states_46 = None
        key_states_47 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = value_states_46.view((1, 2, -1, 96))
        value_states_46 = None
        value_states_47 = view_72.transpose(1, 2)
        view_72 = None
        cos_26 = cos_2.unsqueeze(1)
        sin_26 = sin_2.unsqueeze(1)
        q_rot_23 = query_states_47[(Ellipsis, slice(None, 96, None))]
        q_pass_23 = query_states_47[(Ellipsis, slice(96, None, None))]
        query_states_47 = None
        k_rot_23 = key_states_47[(Ellipsis, slice(None, 96, None))]
        k_pass_23 = key_states_47[(Ellipsis, slice(96, None, None))]
        key_states_47 = None
        mul_212 = q_rot_23 * cos_26
        x1_46 = q_rot_23[(Ellipsis, slice(None, 48, None))]
        x2_46 = q_rot_23[(Ellipsis, slice(48, None, None))]
        q_rot_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_93 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_213 = cat_93 * sin_26
        cat_93 = None
        add_139 = mul_212 + mul_213
        mul_212 = mul_213 = None
        q_embed_23 = torch.cat([add_139, q_pass_23], dim=-1)
        add_139 = q_pass_23 = None
        mul_214 = k_rot_23 * cos_26
        cos_26 = None
        x1_47 = k_rot_23[(Ellipsis, slice(None, 48, None))]
        x2_47 = k_rot_23[(Ellipsis, slice(48, None, None))]
        k_rot_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_95 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_215 = cat_95 * sin_26
        cat_95 = sin_26 = None
        add_140 = mul_214 + mul_215
        mul_214 = mul_215 = None
        k_embed_23 = torch.cat([add_140, k_pass_23], dim=-1)
        add_140 = k_pass_23 = None
        attention_mask_24 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_23 = q_embed_23.contiguous()
        q_embed_23 = None
        key_23 = k_embed_23.contiguous()
        value_23 = value_states_47.contiguous()
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_23,
            value_23,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_23 = key_23 = value_23 = attention_mask_24 = None
        transpose_96 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_96.contiguous()
        transpose_96 = None
        reshape_23 = attn_output_93.reshape(1, 2, -1)
        attn_output_93 = None
        attn_output_94 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_46 = torch.nn.functional.dropout(attn_output_95, 0.0, False, False)
        attn_output_95 = None
        hidden_states_210 = hidden_states_206 + dropout_46
        hidden_states_206 = dropout_46 = None
        hidden_states_211 = hidden_states_210.to(torch.float32)
        pow_48 = hidden_states_211.pow(2)
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_142 = variance_47 + 1e-05
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_212 = hidden_states_211 * rsqrt_47
        hidden_states_211 = rsqrt_47 = None
        to_99 = hidden_states_212.to(torch.bfloat16)
        hidden_states_212 = None
        hidden_states_213 = (
            l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
            * to_99
        )
        l_self_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            to_99
        ) = None
        up_states_69 = torch._C._nn.linear(
            hidden_states_213,
            l_self_modules_layers_modules_23_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_213 = l_self_modules_layers_modules_23_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_23 = up_states_69.chunk(2, dim=-1)
        up_states_69 = None
        gate_23 = chunk_23[0]
        up_states_70 = chunk_23[1]
        chunk_23 = None
        silu_23 = torch.nn.functional.silu(gate_23, inplace=False)
        gate_23 = None
        up_states_71 = up_states_70 * silu_23
        up_states_70 = silu_23 = None
        hidden_states_214 = torch._C._nn.linear(
            up_states_71,
            l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_71 = l_self_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_47 = torch.nn.functional.dropout(hidden_states_214, 0.0, False, False)
        hidden_states_214 = None
        hidden_states_215 = hidden_states_210 + dropout_47
        hidden_states_210 = dropout_47 = None
        hidden_states_216 = hidden_states_215.to(torch.float32)
        pow_49 = hidden_states_216.pow(2)
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_144 = variance_48 + 1e-05
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_217 = hidden_states_216 * rsqrt_48
        hidden_states_216 = rsqrt_48 = None
        to_101 = hidden_states_217.to(torch.bfloat16)
        hidden_states_217 = None
        hidden_states_218 = (
            l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
            * to_101
        )
        l_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            to_101
        ) = None
        qkv_24 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_layers_modules_24_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_218 = l_self_modules_layers_modules_24_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_48 = qkv_24[(Ellipsis, slice(None, 3072, None))]
        key_states_48 = qkv_24[(Ellipsis, slice(3072, 6144, None))]
        value_states_48 = qkv_24[(Ellipsis, slice(6144, None, None))]
        qkv_24 = None
        view_73 = query_states_48.view((1, 2, -1, 96))
        query_states_48 = None
        query_states_49 = view_73.transpose(1, 2)
        view_73 = None
        view_74 = key_states_48.view((1, 2, -1, 96))
        key_states_48 = None
        key_states_49 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = value_states_48.view((1, 2, -1, 96))
        value_states_48 = None
        value_states_49 = view_75.transpose(1, 2)
        view_75 = None
        cos_27 = cos_2.unsqueeze(1)
        sin_27 = sin_2.unsqueeze(1)
        q_rot_24 = query_states_49[(Ellipsis, slice(None, 96, None))]
        q_pass_24 = query_states_49[(Ellipsis, slice(96, None, None))]
        query_states_49 = None
        k_rot_24 = key_states_49[(Ellipsis, slice(None, 96, None))]
        k_pass_24 = key_states_49[(Ellipsis, slice(96, None, None))]
        key_states_49 = None
        mul_221 = q_rot_24 * cos_27
        x1_48 = q_rot_24[(Ellipsis, slice(None, 48, None))]
        x2_48 = q_rot_24[(Ellipsis, slice(48, None, None))]
        q_rot_24 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_97 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_222 = cat_97 * sin_27
        cat_97 = None
        add_145 = mul_221 + mul_222
        mul_221 = mul_222 = None
        q_embed_24 = torch.cat([add_145, q_pass_24], dim=-1)
        add_145 = q_pass_24 = None
        mul_223 = k_rot_24 * cos_27
        cos_27 = None
        x1_49 = k_rot_24[(Ellipsis, slice(None, 48, None))]
        x2_49 = k_rot_24[(Ellipsis, slice(48, None, None))]
        k_rot_24 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_99 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_224 = cat_99 * sin_27
        cat_99 = sin_27 = None
        add_146 = mul_223 + mul_224
        mul_223 = mul_224 = None
        k_embed_24 = torch.cat([add_146, k_pass_24], dim=-1)
        add_146 = k_pass_24 = None
        attention_mask_25 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_24 = q_embed_24.contiguous()
        q_embed_24 = None
        key_24 = k_embed_24.contiguous()
        value_24 = value_states_49.contiguous()
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_24,
            value_24,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_24 = key_24 = value_24 = attention_mask_25 = None
        transpose_100 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_100.contiguous()
        transpose_100 = None
        reshape_24 = attn_output_97.reshape(1, 2, -1)
        attn_output_97 = None
        attn_output_98 = reshape_24.contiguous()
        reshape_24 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_48 = torch.nn.functional.dropout(attn_output_99, 0.0, False, False)
        attn_output_99 = None
        hidden_states_219 = hidden_states_215 + dropout_48
        hidden_states_215 = dropout_48 = None
        hidden_states_220 = hidden_states_219.to(torch.float32)
        pow_50 = hidden_states_220.pow(2)
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_148 = variance_49 + 1e-05
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_148)
        add_148 = None
        hidden_states_221 = hidden_states_220 * rsqrt_49
        hidden_states_220 = rsqrt_49 = None
        to_103 = hidden_states_221.to(torch.bfloat16)
        hidden_states_221 = None
        hidden_states_222 = (
            l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
            * to_103
        )
        l_self_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            to_103
        ) = None
        up_states_72 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_layers_modules_24_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_222 = l_self_modules_layers_modules_24_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_24 = up_states_72.chunk(2, dim=-1)
        up_states_72 = None
        gate_24 = chunk_24[0]
        up_states_73 = chunk_24[1]
        chunk_24 = None
        silu_24 = torch.nn.functional.silu(gate_24, inplace=False)
        gate_24 = None
        up_states_74 = up_states_73 * silu_24
        up_states_73 = silu_24 = None
        hidden_states_223 = torch._C._nn.linear(
            up_states_74,
            l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_74 = l_self_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_49 = torch.nn.functional.dropout(hidden_states_223, 0.0, False, False)
        hidden_states_223 = None
        hidden_states_224 = hidden_states_219 + dropout_49
        hidden_states_219 = dropout_49 = None
        hidden_states_225 = hidden_states_224.to(torch.float32)
        pow_51 = hidden_states_225.pow(2)
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_150 = variance_50 + 1e-05
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_226 = hidden_states_225 * rsqrt_50
        hidden_states_225 = rsqrt_50 = None
        to_105 = hidden_states_226.to(torch.bfloat16)
        hidden_states_226 = None
        hidden_states_227 = (
            l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
            * to_105
        )
        l_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            to_105
        ) = None
        qkv_25 = torch._C._nn.linear(
            hidden_states_227,
            l_self_modules_layers_modules_25_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_227 = l_self_modules_layers_modules_25_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_50 = qkv_25[(Ellipsis, slice(None, 3072, None))]
        key_states_50 = qkv_25[(Ellipsis, slice(3072, 6144, None))]
        value_states_50 = qkv_25[(Ellipsis, slice(6144, None, None))]
        qkv_25 = None
        view_76 = query_states_50.view((1, 2, -1, 96))
        query_states_50 = None
        query_states_51 = view_76.transpose(1, 2)
        view_76 = None
        view_77 = key_states_50.view((1, 2, -1, 96))
        key_states_50 = None
        key_states_51 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = value_states_50.view((1, 2, -1, 96))
        value_states_50 = None
        value_states_51 = view_78.transpose(1, 2)
        view_78 = None
        cos_28 = cos_2.unsqueeze(1)
        sin_28 = sin_2.unsqueeze(1)
        q_rot_25 = query_states_51[(Ellipsis, slice(None, 96, None))]
        q_pass_25 = query_states_51[(Ellipsis, slice(96, None, None))]
        query_states_51 = None
        k_rot_25 = key_states_51[(Ellipsis, slice(None, 96, None))]
        k_pass_25 = key_states_51[(Ellipsis, slice(96, None, None))]
        key_states_51 = None
        mul_230 = q_rot_25 * cos_28
        x1_50 = q_rot_25[(Ellipsis, slice(None, 48, None))]
        x2_50 = q_rot_25[(Ellipsis, slice(48, None, None))]
        q_rot_25 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_101 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_231 = cat_101 * sin_28
        cat_101 = None
        add_151 = mul_230 + mul_231
        mul_230 = mul_231 = None
        q_embed_25 = torch.cat([add_151, q_pass_25], dim=-1)
        add_151 = q_pass_25 = None
        mul_232 = k_rot_25 * cos_28
        cos_28 = None
        x1_51 = k_rot_25[(Ellipsis, slice(None, 48, None))]
        x2_51 = k_rot_25[(Ellipsis, slice(48, None, None))]
        k_rot_25 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_103 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_233 = cat_103 * sin_28
        cat_103 = sin_28 = None
        add_152 = mul_232 + mul_233
        mul_232 = mul_233 = None
        k_embed_25 = torch.cat([add_152, k_pass_25], dim=-1)
        add_152 = k_pass_25 = None
        attention_mask_26 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_25 = q_embed_25.contiguous()
        q_embed_25 = None
        key_25 = k_embed_25.contiguous()
        value_25 = value_states_51.contiguous()
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_25,
            value_25,
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_25 = key_25 = value_25 = attention_mask_26 = None
        transpose_104 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_104.contiguous()
        transpose_104 = None
        reshape_25 = attn_output_101.reshape(1, 2, -1)
        attn_output_101 = None
        attn_output_102 = reshape_25.contiguous()
        reshape_25 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_50 = torch.nn.functional.dropout(attn_output_103, 0.0, False, False)
        attn_output_103 = None
        hidden_states_228 = hidden_states_224 + dropout_50
        hidden_states_224 = dropout_50 = None
        hidden_states_229 = hidden_states_228.to(torch.float32)
        pow_52 = hidden_states_229.pow(2)
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_154 = variance_51 + 1e-05
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_154)
        add_154 = None
        hidden_states_230 = hidden_states_229 * rsqrt_51
        hidden_states_229 = rsqrt_51 = None
        to_107 = hidden_states_230.to(torch.bfloat16)
        hidden_states_230 = None
        hidden_states_231 = (
            l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
            * to_107
        )
        l_self_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            to_107
        ) = None
        up_states_75 = torch._C._nn.linear(
            hidden_states_231,
            l_self_modules_layers_modules_25_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_231 = l_self_modules_layers_modules_25_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_25 = up_states_75.chunk(2, dim=-1)
        up_states_75 = None
        gate_25 = chunk_25[0]
        up_states_76 = chunk_25[1]
        chunk_25 = None
        silu_25 = torch.nn.functional.silu(gate_25, inplace=False)
        gate_25 = None
        up_states_77 = up_states_76 * silu_25
        up_states_76 = silu_25 = None
        hidden_states_232 = torch._C._nn.linear(
            up_states_77,
            l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_77 = l_self_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_51 = torch.nn.functional.dropout(hidden_states_232, 0.0, False, False)
        hidden_states_232 = None
        hidden_states_233 = hidden_states_228 + dropout_51
        hidden_states_228 = dropout_51 = None
        hidden_states_234 = hidden_states_233.to(torch.float32)
        pow_53 = hidden_states_234.pow(2)
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_156 = variance_52 + 1e-05
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_156)
        add_156 = None
        hidden_states_235 = hidden_states_234 * rsqrt_52
        hidden_states_234 = rsqrt_52 = None
        to_109 = hidden_states_235.to(torch.bfloat16)
        hidden_states_235 = None
        hidden_states_236 = (
            l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
            * to_109
        )
        l_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            to_109
        ) = None
        qkv_26 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_layers_modules_26_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_236 = l_self_modules_layers_modules_26_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_52 = qkv_26[(Ellipsis, slice(None, 3072, None))]
        key_states_52 = qkv_26[(Ellipsis, slice(3072, 6144, None))]
        value_states_52 = qkv_26[(Ellipsis, slice(6144, None, None))]
        qkv_26 = None
        view_79 = query_states_52.view((1, 2, -1, 96))
        query_states_52 = None
        query_states_53 = view_79.transpose(1, 2)
        view_79 = None
        view_80 = key_states_52.view((1, 2, -1, 96))
        key_states_52 = None
        key_states_53 = view_80.transpose(1, 2)
        view_80 = None
        view_81 = value_states_52.view((1, 2, -1, 96))
        value_states_52 = None
        value_states_53 = view_81.transpose(1, 2)
        view_81 = None
        cos_29 = cos_2.unsqueeze(1)
        sin_29 = sin_2.unsqueeze(1)
        q_rot_26 = query_states_53[(Ellipsis, slice(None, 96, None))]
        q_pass_26 = query_states_53[(Ellipsis, slice(96, None, None))]
        query_states_53 = None
        k_rot_26 = key_states_53[(Ellipsis, slice(None, 96, None))]
        k_pass_26 = key_states_53[(Ellipsis, slice(96, None, None))]
        key_states_53 = None
        mul_239 = q_rot_26 * cos_29
        x1_52 = q_rot_26[(Ellipsis, slice(None, 48, None))]
        x2_52 = q_rot_26[(Ellipsis, slice(48, None, None))]
        q_rot_26 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_105 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_240 = cat_105 * sin_29
        cat_105 = None
        add_157 = mul_239 + mul_240
        mul_239 = mul_240 = None
        q_embed_26 = torch.cat([add_157, q_pass_26], dim=-1)
        add_157 = q_pass_26 = None
        mul_241 = k_rot_26 * cos_29
        cos_29 = None
        x1_53 = k_rot_26[(Ellipsis, slice(None, 48, None))]
        x2_53 = k_rot_26[(Ellipsis, slice(48, None, None))]
        k_rot_26 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_107 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_242 = cat_107 * sin_29
        cat_107 = sin_29 = None
        add_158 = mul_241 + mul_242
        mul_241 = mul_242 = None
        k_embed_26 = torch.cat([add_158, k_pass_26], dim=-1)
        add_158 = k_pass_26 = None
        attention_mask_27 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_26 = q_embed_26.contiguous()
        q_embed_26 = None
        key_26 = k_embed_26.contiguous()
        value_26 = value_states_53.contiguous()
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_26,
            value_26,
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_26 = key_26 = value_26 = attention_mask_27 = None
        transpose_108 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_108.contiguous()
        transpose_108 = None
        reshape_26 = attn_output_105.reshape(1, 2, -1)
        attn_output_105 = None
        attn_output_106 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_52 = torch.nn.functional.dropout(attn_output_107, 0.0, False, False)
        attn_output_107 = None
        hidden_states_237 = hidden_states_233 + dropout_52
        hidden_states_233 = dropout_52 = None
        hidden_states_238 = hidden_states_237.to(torch.float32)
        pow_54 = hidden_states_238.pow(2)
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_160 = variance_53 + 1e-05
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_239 = hidden_states_238 * rsqrt_53
        hidden_states_238 = rsqrt_53 = None
        to_111 = hidden_states_239.to(torch.bfloat16)
        hidden_states_239 = None
        hidden_states_240 = (
            l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
            * to_111
        )
        l_self_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = (
            to_111
        ) = None
        up_states_78 = torch._C._nn.linear(
            hidden_states_240,
            l_self_modules_layers_modules_26_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_240 = l_self_modules_layers_modules_26_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_26 = up_states_78.chunk(2, dim=-1)
        up_states_78 = None
        gate_26 = chunk_26[0]
        up_states_79 = chunk_26[1]
        chunk_26 = None
        silu_26 = torch.nn.functional.silu(gate_26, inplace=False)
        gate_26 = None
        up_states_80 = up_states_79 * silu_26
        up_states_79 = silu_26 = None
        hidden_states_241 = torch._C._nn.linear(
            up_states_80,
            l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_80 = l_self_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_53 = torch.nn.functional.dropout(hidden_states_241, 0.0, False, False)
        hidden_states_241 = None
        hidden_states_242 = hidden_states_237 + dropout_53
        hidden_states_237 = dropout_53 = None
        hidden_states_243 = hidden_states_242.to(torch.float32)
        pow_55 = hidden_states_243.pow(2)
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_162 = variance_54 + 1e-05
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_162)
        add_162 = None
        hidden_states_244 = hidden_states_243 * rsqrt_54
        hidden_states_243 = rsqrt_54 = None
        to_113 = hidden_states_244.to(torch.bfloat16)
        hidden_states_244 = None
        hidden_states_245 = (
            l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
            * to_113
        )
        l_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            to_113
        ) = None
        qkv_27 = torch._C._nn.linear(
            hidden_states_245,
            l_self_modules_layers_modules_27_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_245 = l_self_modules_layers_modules_27_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_54 = qkv_27[(Ellipsis, slice(None, 3072, None))]
        key_states_54 = qkv_27[(Ellipsis, slice(3072, 6144, None))]
        value_states_54 = qkv_27[(Ellipsis, slice(6144, None, None))]
        qkv_27 = None
        view_82 = query_states_54.view((1, 2, -1, 96))
        query_states_54 = None
        query_states_55 = view_82.transpose(1, 2)
        view_82 = None
        view_83 = key_states_54.view((1, 2, -1, 96))
        key_states_54 = None
        key_states_55 = view_83.transpose(1, 2)
        view_83 = None
        view_84 = value_states_54.view((1, 2, -1, 96))
        value_states_54 = None
        value_states_55 = view_84.transpose(1, 2)
        view_84 = None
        cos_30 = cos_2.unsqueeze(1)
        sin_30 = sin_2.unsqueeze(1)
        q_rot_27 = query_states_55[(Ellipsis, slice(None, 96, None))]
        q_pass_27 = query_states_55[(Ellipsis, slice(96, None, None))]
        query_states_55 = None
        k_rot_27 = key_states_55[(Ellipsis, slice(None, 96, None))]
        k_pass_27 = key_states_55[(Ellipsis, slice(96, None, None))]
        key_states_55 = None
        mul_248 = q_rot_27 * cos_30
        x1_54 = q_rot_27[(Ellipsis, slice(None, 48, None))]
        x2_54 = q_rot_27[(Ellipsis, slice(48, None, None))]
        q_rot_27 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_109 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_249 = cat_109 * sin_30
        cat_109 = None
        add_163 = mul_248 + mul_249
        mul_248 = mul_249 = None
        q_embed_27 = torch.cat([add_163, q_pass_27], dim=-1)
        add_163 = q_pass_27 = None
        mul_250 = k_rot_27 * cos_30
        cos_30 = None
        x1_55 = k_rot_27[(Ellipsis, slice(None, 48, None))]
        x2_55 = k_rot_27[(Ellipsis, slice(48, None, None))]
        k_rot_27 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_111 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_251 = cat_111 * sin_30
        cat_111 = sin_30 = None
        add_164 = mul_250 + mul_251
        mul_250 = mul_251 = None
        k_embed_27 = torch.cat([add_164, k_pass_27], dim=-1)
        add_164 = k_pass_27 = None
        attention_mask_28 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_27 = q_embed_27.contiguous()
        q_embed_27 = None
        key_27 = k_embed_27.contiguous()
        value_27 = value_states_55.contiguous()
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_27,
            value_27,
            attn_mask=attention_mask_28,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_27 = key_27 = value_27 = attention_mask_28 = None
        transpose_112 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_112.contiguous()
        transpose_112 = None
        reshape_27 = attn_output_109.reshape(1, 2, -1)
        attn_output_109 = None
        attn_output_110 = reshape_27.contiguous()
        reshape_27 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_54 = torch.nn.functional.dropout(attn_output_111, 0.0, False, False)
        attn_output_111 = None
        hidden_states_246 = hidden_states_242 + dropout_54
        hidden_states_242 = dropout_54 = None
        hidden_states_247 = hidden_states_246.to(torch.float32)
        pow_56 = hidden_states_247.pow(2)
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_166 = variance_55 + 1e-05
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_248 = hidden_states_247 * rsqrt_55
        hidden_states_247 = rsqrt_55 = None
        to_115 = hidden_states_248.to(torch.bfloat16)
        hidden_states_248 = None
        hidden_states_249 = (
            l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
            * to_115
        )
        l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = (
            to_115
        ) = None
        up_states_81 = torch._C._nn.linear(
            hidden_states_249,
            l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_249 = l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_27 = up_states_81.chunk(2, dim=-1)
        up_states_81 = None
        gate_27 = chunk_27[0]
        up_states_82 = chunk_27[1]
        chunk_27 = None
        silu_27 = torch.nn.functional.silu(gate_27, inplace=False)
        gate_27 = None
        up_states_83 = up_states_82 * silu_27
        up_states_82 = silu_27 = None
        hidden_states_250 = torch._C._nn.linear(
            up_states_83,
            l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_83 = l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_55 = torch.nn.functional.dropout(hidden_states_250, 0.0, False, False)
        hidden_states_250 = None
        hidden_states_251 = hidden_states_246 + dropout_55
        hidden_states_246 = dropout_55 = None
        hidden_states_252 = hidden_states_251.to(torch.float32)
        pow_57 = hidden_states_252.pow(2)
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_168 = variance_56 + 1e-05
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_253 = hidden_states_252 * rsqrt_56
        hidden_states_252 = rsqrt_56 = None
        to_117 = hidden_states_253.to(torch.bfloat16)
        hidden_states_253 = None
        hidden_states_254 = (
            l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
            * to_117
        )
        l_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = (
            to_117
        ) = None
        qkv_28 = torch._C._nn.linear(
            hidden_states_254,
            l_self_modules_layers_modules_28_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_254 = l_self_modules_layers_modules_28_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_56 = qkv_28[(Ellipsis, slice(None, 3072, None))]
        key_states_56 = qkv_28[(Ellipsis, slice(3072, 6144, None))]
        value_states_56 = qkv_28[(Ellipsis, slice(6144, None, None))]
        qkv_28 = None
        view_85 = query_states_56.view((1, 2, -1, 96))
        query_states_56 = None
        query_states_57 = view_85.transpose(1, 2)
        view_85 = None
        view_86 = key_states_56.view((1, 2, -1, 96))
        key_states_56 = None
        key_states_57 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = value_states_56.view((1, 2, -1, 96))
        value_states_56 = None
        value_states_57 = view_87.transpose(1, 2)
        view_87 = None
        cos_31 = cos_2.unsqueeze(1)
        sin_31 = sin_2.unsqueeze(1)
        q_rot_28 = query_states_57[(Ellipsis, slice(None, 96, None))]
        q_pass_28 = query_states_57[(Ellipsis, slice(96, None, None))]
        query_states_57 = None
        k_rot_28 = key_states_57[(Ellipsis, slice(None, 96, None))]
        k_pass_28 = key_states_57[(Ellipsis, slice(96, None, None))]
        key_states_57 = None
        mul_257 = q_rot_28 * cos_31
        x1_56 = q_rot_28[(Ellipsis, slice(None, 48, None))]
        x2_56 = q_rot_28[(Ellipsis, slice(48, None, None))]
        q_rot_28 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_113 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_258 = cat_113 * sin_31
        cat_113 = None
        add_169 = mul_257 + mul_258
        mul_257 = mul_258 = None
        q_embed_28 = torch.cat([add_169, q_pass_28], dim=-1)
        add_169 = q_pass_28 = None
        mul_259 = k_rot_28 * cos_31
        cos_31 = None
        x1_57 = k_rot_28[(Ellipsis, slice(None, 48, None))]
        x2_57 = k_rot_28[(Ellipsis, slice(48, None, None))]
        k_rot_28 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_115 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_260 = cat_115 * sin_31
        cat_115 = sin_31 = None
        add_170 = mul_259 + mul_260
        mul_259 = mul_260 = None
        k_embed_28 = torch.cat([add_170, k_pass_28], dim=-1)
        add_170 = k_pass_28 = None
        attention_mask_29 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_28 = q_embed_28.contiguous()
        q_embed_28 = None
        key_28 = k_embed_28.contiguous()
        value_28 = value_states_57.contiguous()
        attn_output_112 = torch._C._nn.scaled_dot_product_attention(
            query_28,
            key_28,
            value_28,
            attn_mask=attention_mask_29,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_28 = key_28 = value_28 = attention_mask_29 = None
        transpose_116 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_116.contiguous()
        transpose_116 = None
        reshape_28 = attn_output_113.reshape(1, 2, -1)
        attn_output_113 = None
        attn_output_114 = reshape_28.contiguous()
        reshape_28 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_56 = torch.nn.functional.dropout(attn_output_115, 0.0, False, False)
        attn_output_115 = None
        hidden_states_255 = hidden_states_251 + dropout_56
        hidden_states_251 = dropout_56 = None
        hidden_states_256 = hidden_states_255.to(torch.float32)
        pow_58 = hidden_states_256.pow(2)
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_172 = variance_57 + 1e-05
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_172)
        add_172 = None
        hidden_states_257 = hidden_states_256 * rsqrt_57
        hidden_states_256 = rsqrt_57 = None
        to_119 = hidden_states_257.to(torch.bfloat16)
        hidden_states_257 = None
        hidden_states_258 = (
            l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
            * to_119
        )
        l_self_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = (
            to_119
        ) = None
        up_states_84 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_layers_modules_28_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_258 = l_self_modules_layers_modules_28_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_28 = up_states_84.chunk(2, dim=-1)
        up_states_84 = None
        gate_28 = chunk_28[0]
        up_states_85 = chunk_28[1]
        chunk_28 = None
        silu_28 = torch.nn.functional.silu(gate_28, inplace=False)
        gate_28 = None
        up_states_86 = up_states_85 * silu_28
        up_states_85 = silu_28 = None
        hidden_states_259 = torch._C._nn.linear(
            up_states_86,
            l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_86 = l_self_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_57 = torch.nn.functional.dropout(hidden_states_259, 0.0, False, False)
        hidden_states_259 = None
        hidden_states_260 = hidden_states_255 + dropout_57
        hidden_states_255 = dropout_57 = None
        hidden_states_261 = hidden_states_260.to(torch.float32)
        pow_59 = hidden_states_261.pow(2)
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_174 = variance_58 + 1e-05
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_262 = hidden_states_261 * rsqrt_58
        hidden_states_261 = rsqrt_58 = None
        to_121 = hidden_states_262.to(torch.bfloat16)
        hidden_states_262 = None
        hidden_states_263 = (
            l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
            * to_121
        )
        l_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = (
            to_121
        ) = None
        qkv_29 = torch._C._nn.linear(
            hidden_states_263,
            l_self_modules_layers_modules_29_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_263 = l_self_modules_layers_modules_29_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_58 = qkv_29[(Ellipsis, slice(None, 3072, None))]
        key_states_58 = qkv_29[(Ellipsis, slice(3072, 6144, None))]
        value_states_58 = qkv_29[(Ellipsis, slice(6144, None, None))]
        qkv_29 = None
        view_88 = query_states_58.view((1, 2, -1, 96))
        query_states_58 = None
        query_states_59 = view_88.transpose(1, 2)
        view_88 = None
        view_89 = key_states_58.view((1, 2, -1, 96))
        key_states_58 = None
        key_states_59 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = value_states_58.view((1, 2, -1, 96))
        value_states_58 = None
        value_states_59 = view_90.transpose(1, 2)
        view_90 = None
        cos_32 = cos_2.unsqueeze(1)
        sin_32 = sin_2.unsqueeze(1)
        q_rot_29 = query_states_59[(Ellipsis, slice(None, 96, None))]
        q_pass_29 = query_states_59[(Ellipsis, slice(96, None, None))]
        query_states_59 = None
        k_rot_29 = key_states_59[(Ellipsis, slice(None, 96, None))]
        k_pass_29 = key_states_59[(Ellipsis, slice(96, None, None))]
        key_states_59 = None
        mul_266 = q_rot_29 * cos_32
        x1_58 = q_rot_29[(Ellipsis, slice(None, 48, None))]
        x2_58 = q_rot_29[(Ellipsis, slice(48, None, None))]
        q_rot_29 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_117 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_267 = cat_117 * sin_32
        cat_117 = None
        add_175 = mul_266 + mul_267
        mul_266 = mul_267 = None
        q_embed_29 = torch.cat([add_175, q_pass_29], dim=-1)
        add_175 = q_pass_29 = None
        mul_268 = k_rot_29 * cos_32
        cos_32 = None
        x1_59 = k_rot_29[(Ellipsis, slice(None, 48, None))]
        x2_59 = k_rot_29[(Ellipsis, slice(48, None, None))]
        k_rot_29 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_119 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_269 = cat_119 * sin_32
        cat_119 = sin_32 = None
        add_176 = mul_268 + mul_269
        mul_268 = mul_269 = None
        k_embed_29 = torch.cat([add_176, k_pass_29], dim=-1)
        add_176 = k_pass_29 = None
        attention_mask_30 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_29 = q_embed_29.contiguous()
        q_embed_29 = None
        key_29 = k_embed_29.contiguous()
        value_29 = value_states_59.contiguous()
        attn_output_116 = torch._C._nn.scaled_dot_product_attention(
            query_29,
            key_29,
            value_29,
            attn_mask=attention_mask_30,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_29 = key_29 = value_29 = attention_mask_30 = None
        transpose_120 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_120.contiguous()
        transpose_120 = None
        reshape_29 = attn_output_117.reshape(1, 2, -1)
        attn_output_117 = None
        attn_output_118 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_58 = torch.nn.functional.dropout(attn_output_119, 0.0, False, False)
        attn_output_119 = None
        hidden_states_264 = hidden_states_260 + dropout_58
        hidden_states_260 = dropout_58 = None
        hidden_states_265 = hidden_states_264.to(torch.float32)
        pow_60 = hidden_states_265.pow(2)
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_178 = variance_59 + 1e-05
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_178)
        add_178 = None
        hidden_states_266 = hidden_states_265 * rsqrt_59
        hidden_states_265 = rsqrt_59 = None
        to_123 = hidden_states_266.to(torch.bfloat16)
        hidden_states_266 = None
        hidden_states_267 = (
            l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
            * to_123
        )
        l_self_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = (
            to_123
        ) = None
        up_states_87 = torch._C._nn.linear(
            hidden_states_267,
            l_self_modules_layers_modules_29_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_267 = l_self_modules_layers_modules_29_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_29 = up_states_87.chunk(2, dim=-1)
        up_states_87 = None
        gate_29 = chunk_29[0]
        up_states_88 = chunk_29[1]
        chunk_29 = None
        silu_29 = torch.nn.functional.silu(gate_29, inplace=False)
        gate_29 = None
        up_states_89 = up_states_88 * silu_29
        up_states_88 = silu_29 = None
        hidden_states_268 = torch._C._nn.linear(
            up_states_89,
            l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_89 = l_self_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_59 = torch.nn.functional.dropout(hidden_states_268, 0.0, False, False)
        hidden_states_268 = None
        hidden_states_269 = hidden_states_264 + dropout_59
        hidden_states_264 = dropout_59 = None
        hidden_states_270 = hidden_states_269.to(torch.float32)
        pow_61 = hidden_states_270.pow(2)
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_180 = variance_60 + 1e-05
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_180)
        add_180 = None
        hidden_states_271 = hidden_states_270 * rsqrt_60
        hidden_states_270 = rsqrt_60 = None
        to_125 = hidden_states_271.to(torch.bfloat16)
        hidden_states_271 = None
        hidden_states_272 = (
            l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
            * to_125
        )
        l_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = (
            to_125
        ) = None
        qkv_30 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_layers_modules_30_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_272 = l_self_modules_layers_modules_30_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_60 = qkv_30[(Ellipsis, slice(None, 3072, None))]
        key_states_60 = qkv_30[(Ellipsis, slice(3072, 6144, None))]
        value_states_60 = qkv_30[(Ellipsis, slice(6144, None, None))]
        qkv_30 = None
        view_91 = query_states_60.view((1, 2, -1, 96))
        query_states_60 = None
        query_states_61 = view_91.transpose(1, 2)
        view_91 = None
        view_92 = key_states_60.view((1, 2, -1, 96))
        key_states_60 = None
        key_states_61 = view_92.transpose(1, 2)
        view_92 = None
        view_93 = value_states_60.view((1, 2, -1, 96))
        value_states_60 = None
        value_states_61 = view_93.transpose(1, 2)
        view_93 = None
        cos_33 = cos_2.unsqueeze(1)
        sin_33 = sin_2.unsqueeze(1)
        q_rot_30 = query_states_61[(Ellipsis, slice(None, 96, None))]
        q_pass_30 = query_states_61[(Ellipsis, slice(96, None, None))]
        query_states_61 = None
        k_rot_30 = key_states_61[(Ellipsis, slice(None, 96, None))]
        k_pass_30 = key_states_61[(Ellipsis, slice(96, None, None))]
        key_states_61 = None
        mul_275 = q_rot_30 * cos_33
        x1_60 = q_rot_30[(Ellipsis, slice(None, 48, None))]
        x2_60 = q_rot_30[(Ellipsis, slice(48, None, None))]
        q_rot_30 = None
        neg_60 = -x2_60
        x2_60 = None
        cat_121 = torch.cat((neg_60, x1_60), dim=-1)
        neg_60 = x1_60 = None
        mul_276 = cat_121 * sin_33
        cat_121 = None
        add_181 = mul_275 + mul_276
        mul_275 = mul_276 = None
        q_embed_30 = torch.cat([add_181, q_pass_30], dim=-1)
        add_181 = q_pass_30 = None
        mul_277 = k_rot_30 * cos_33
        cos_33 = None
        x1_61 = k_rot_30[(Ellipsis, slice(None, 48, None))]
        x2_61 = k_rot_30[(Ellipsis, slice(48, None, None))]
        k_rot_30 = None
        neg_61 = -x2_61
        x2_61 = None
        cat_123 = torch.cat((neg_61, x1_61), dim=-1)
        neg_61 = x1_61 = None
        mul_278 = cat_123 * sin_33
        cat_123 = sin_33 = None
        add_182 = mul_277 + mul_278
        mul_277 = mul_278 = None
        k_embed_30 = torch.cat([add_182, k_pass_30], dim=-1)
        add_182 = k_pass_30 = None
        attention_mask_31 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        query_30 = q_embed_30.contiguous()
        q_embed_30 = None
        key_30 = k_embed_30.contiguous()
        value_30 = value_states_61.contiguous()
        attn_output_120 = torch._C._nn.scaled_dot_product_attention(
            query_30,
            key_30,
            value_30,
            attn_mask=attention_mask_31,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_30 = key_30 = value_30 = attention_mask_31 = None
        transpose_124 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_124.contiguous()
        transpose_124 = None
        reshape_30 = attn_output_121.reshape(1, 2, -1)
        attn_output_121 = None
        attn_output_122 = reshape_30.contiguous()
        reshape_30 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_60 = torch.nn.functional.dropout(attn_output_123, 0.0, False, False)
        attn_output_123 = None
        hidden_states_273 = hidden_states_269 + dropout_60
        hidden_states_269 = dropout_60 = None
        hidden_states_274 = hidden_states_273.to(torch.float32)
        pow_62 = hidden_states_274.pow(2)
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_184 = variance_61 + 1e-05
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_275 = hidden_states_274 * rsqrt_61
        hidden_states_274 = rsqrt_61 = None
        to_127 = hidden_states_275.to(torch.bfloat16)
        hidden_states_275 = None
        hidden_states_276 = (
            l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
            * to_127
        )
        l_self_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = (
            to_127
        ) = None
        up_states_90 = torch._C._nn.linear(
            hidden_states_276,
            l_self_modules_layers_modules_30_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_276 = l_self_modules_layers_modules_30_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_30 = up_states_90.chunk(2, dim=-1)
        up_states_90 = None
        gate_30 = chunk_30[0]
        up_states_91 = chunk_30[1]
        chunk_30 = None
        silu_30 = torch.nn.functional.silu(gate_30, inplace=False)
        gate_30 = None
        up_states_92 = up_states_91 * silu_30
        up_states_91 = silu_30 = None
        hidden_states_277 = torch._C._nn.linear(
            up_states_92,
            l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_92 = l_self_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_61 = torch.nn.functional.dropout(hidden_states_277, 0.0, False, False)
        hidden_states_277 = None
        hidden_states_278 = hidden_states_273 + dropout_61
        hidden_states_273 = dropout_61 = None
        hidden_states_279 = hidden_states_278.to(torch.float32)
        pow_63 = hidden_states_279.pow(2)
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_186 = variance_62 + 1e-05
        variance_62 = None
        rsqrt_62 = torch.rsqrt(add_186)
        add_186 = None
        hidden_states_280 = hidden_states_279 * rsqrt_62
        hidden_states_279 = rsqrt_62 = None
        to_129 = hidden_states_280.to(torch.bfloat16)
        hidden_states_280 = None
        hidden_states_281 = (
            l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
            * to_129
        )
        l_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = (
            to_129
        ) = None
        qkv_31 = torch._C._nn.linear(
            hidden_states_281,
            l_self_modules_layers_modules_31_modules_self_attn_modules_qkv_proj_parameters_weight_,
            None,
        )
        hidden_states_281 = l_self_modules_layers_modules_31_modules_self_attn_modules_qkv_proj_parameters_weight_ = (None)
        query_states_62 = qkv_31[(Ellipsis, slice(None, 3072, None))]
        key_states_62 = qkv_31[(Ellipsis, slice(3072, 6144, None))]
        value_states_62 = qkv_31[(Ellipsis, slice(6144, None, None))]
        qkv_31 = None
        view_94 = query_states_62.view((1, 2, -1, 96))
        query_states_62 = None
        query_states_63 = view_94.transpose(1, 2)
        view_94 = None
        view_95 = key_states_62.view((1, 2, -1, 96))
        key_states_62 = None
        key_states_63 = view_95.transpose(1, 2)
        view_95 = None
        view_96 = value_states_62.view((1, 2, -1, 96))
        value_states_62 = None
        value_states_63 = view_96.transpose(1, 2)
        view_96 = None
        cos_34 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_34 = sin_2.unsqueeze(1)
        sin_2 = None
        q_rot_31 = query_states_63[(Ellipsis, slice(None, 96, None))]
        q_pass_31 = query_states_63[(Ellipsis, slice(96, None, None))]
        query_states_63 = None
        k_rot_31 = key_states_63[(Ellipsis, slice(None, 96, None))]
        k_pass_31 = key_states_63[(Ellipsis, slice(96, None, None))]
        key_states_63 = None
        mul_284 = q_rot_31 * cos_34
        x1_62 = q_rot_31[(Ellipsis, slice(None, 48, None))]
        x2_62 = q_rot_31[(Ellipsis, slice(48, None, None))]
        q_rot_31 = None
        neg_62 = -x2_62
        x2_62 = None
        cat_125 = torch.cat((neg_62, x1_62), dim=-1)
        neg_62 = x1_62 = None
        mul_285 = cat_125 * sin_34
        cat_125 = None
        add_187 = mul_284 + mul_285
        mul_284 = mul_285 = None
        q_embed_31 = torch.cat([add_187, q_pass_31], dim=-1)
        add_187 = q_pass_31 = None
        mul_286 = k_rot_31 * cos_34
        cos_34 = None
        x1_63 = k_rot_31[(Ellipsis, slice(None, 48, None))]
        x2_63 = k_rot_31[(Ellipsis, slice(48, None, None))]
        k_rot_31 = None
        neg_63 = -x2_63
        x2_63 = None
        cat_127 = torch.cat((neg_63, x1_63), dim=-1)
        neg_63 = x1_63 = None
        mul_287 = cat_127 * sin_34
        cat_127 = sin_34 = None
        add_188 = mul_286 + mul_287
        mul_286 = mul_287 = None
        k_embed_31 = torch.cat([add_188, k_pass_31], dim=-1)
        add_188 = k_pass_31 = None
        attention_mask_32 = causal_mask_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 2, None),
            )
        ]
        causal_mask_3 = None
        query_31 = q_embed_31.contiguous()
        q_embed_31 = None
        key_31 = k_embed_31.contiguous()
        value_31 = value_states_63.contiguous()
        attn_output_124 = torch._C._nn.scaled_dot_product_attention(
            query_31,
            key_31,
            value_31,
            attn_mask=attention_mask_32,
            dropout_p=0.0,
            scale=0.10206207261596575,
            is_causal=False,
        )
        query_31 = key_31 = value_31 = attention_mask_32 = None
        transpose_128 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_128.contiguous()
        transpose_128 = None
        reshape_31 = attn_output_125.reshape(1, 2, -1)
        attn_output_125 = None
        attn_output_126 = reshape_31.contiguous()
        reshape_31 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        dropout_62 = torch.nn.functional.dropout(attn_output_127, 0.0, False, False)
        attn_output_127 = None
        hidden_states_282 = hidden_states_278 + dropout_62
        hidden_states_278 = dropout_62 = None
        hidden_states_283 = hidden_states_282.to(torch.float32)
        pow_64 = hidden_states_283.pow(2)
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_190 = variance_63 + 1e-05
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_284 = hidden_states_283 * rsqrt_63
        hidden_states_283 = rsqrt_63 = None
        to_131 = hidden_states_284.to(torch.bfloat16)
        hidden_states_284 = None
        hidden_states_285 = (
            l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
            * to_131
        )
        l_self_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = (
            to_131
        ) = None
        up_states_93 = torch._C._nn.linear(
            hidden_states_285,
            l_self_modules_layers_modules_31_modules_mlp_modules_gate_up_proj_parameters_weight_,
            None,
        )
        hidden_states_285 = l_self_modules_layers_modules_31_modules_mlp_modules_gate_up_proj_parameters_weight_ = (None)
        chunk_31 = up_states_93.chunk(2, dim=-1)
        up_states_93 = None
        gate_31 = chunk_31[0]
        up_states_94 = chunk_31[1]
        chunk_31 = None
        silu_31 = torch.nn.functional.silu(gate_31, inplace=False)
        gate_31 = None
        up_states_95 = up_states_94 * silu_31
        up_states_94 = silu_31 = None
        hidden_states_286 = torch._C._nn.linear(
            up_states_95,
            l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        up_states_95 = l_self_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        dropout_63 = torch.nn.functional.dropout(hidden_states_286, 0.0, False, False)
        hidden_states_286 = None
        hidden_states_287 = hidden_states_282 + dropout_63
        hidden_states_282 = dropout_63 = None
        hidden_states_288 = hidden_states_287.to(torch.float32)
        hidden_states_287 = None
        pow_65 = hidden_states_288.pow(2)
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_192 = variance_64 + 1e-05
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_289 = hidden_states_288 * rsqrt_64
        hidden_states_288 = rsqrt_64 = None
        to_133 = hidden_states_289.to(torch.bfloat16)
        hidden_states_289 = None
        hidden_states_290 = l_self_modules_norm_parameters_weight_ * to_133
        l_self_modules_norm_parameters_weight_ = to_133 = None
        return (
            value_states_1,
            k_embed,
            value_states_3,
            k_embed_1,
            value_states_5,
            k_embed_2,
            value_states_7,
            k_embed_3,
            value_states_9,
            k_embed_4,
            value_states_11,
            k_embed_5,
            value_states_13,
            k_embed_6,
            value_states_15,
            k_embed_7,
            value_states_17,
            k_embed_8,
            value_states_19,
            k_embed_9,
            value_states_21,
            k_embed_10,
            value_states_23,
            k_embed_11,
            value_states_25,
            k_embed_12,
            value_states_27,
            k_embed_13,
            value_states_29,
            k_embed_14,
            value_states_31,
            k_embed_15,
            value_states_33,
            k_embed_16,
            value_states_35,
            k_embed_17,
            value_states_37,
            k_embed_18,
            value_states_39,
            k_embed_19,
            value_states_41,
            k_embed_20,
            value_states_43,
            k_embed_21,
            value_states_45,
            k_embed_22,
            value_states_47,
            k_embed_23,
            value_states_49,
            k_embed_24,
            value_states_51,
            k_embed_25,
            value_states_53,
            k_embed_26,
            value_states_55,
            k_embed_27,
            value_states_57,
            k_embed_28,
            value_states_59,
            k_embed_29,
            value_states_61,
            k_embed_30,
            value_states_63,
            k_embed_31,
            hidden_states_290,
        )
