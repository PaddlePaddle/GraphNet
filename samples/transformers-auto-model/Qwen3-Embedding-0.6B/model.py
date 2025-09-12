import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_model_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_embed_tokens_parameters_weight_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_model_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_model_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = None
        cache_position = torch.arange(0, 20, device=device(type="cuda", index=0))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cuda", index=0), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(20, device=device(type="cuda", index=0))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cuda", index=0))
        head_arange = torch.arange(1, device=device(type="cuda", index=0))
        lazy_load_decompositions = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions = None
        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(
            1, "error"
        )
        _vmap_increment_nesting = None
        child = torch._functorch.predispatch._add_batch_dim(batch_arange, 0, 1)
        batch_arange = None
        lazy_load_decompositions_1 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_1 = None
        _vmap_increment_nesting_1 = (
            torch._functorch.predispatch._vmap_increment_nesting(1, "error")
        )
        _vmap_increment_nesting_1 = None
        child_1 = torch._functorch.predispatch._add_batch_dim(head_arange, 0, 2)
        head_arange = child_1 = None
        lazy_load_decompositions_2 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_2 = None
        _vmap_increment_nesting_2 = (
            torch._functorch.predispatch._vmap_increment_nesting(20, "error")
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._functorch.predispatch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = (
            torch._functorch.predispatch.lazy_load_decompositions()
        )
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = (
            torch._functorch.predispatch._vmap_increment_nesting(20, "error")
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._functorch.predispatch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        to_1 = le.to(device(type="cuda", index=0))
        le = None
        result_1 = result.__and__(to_1)
        result = to_1 = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        to_2 = index.to(device(type="cuda", index=0))
        index = None
        result_2 = result_1.__and__(to_2)
        result_1 = to_2 = None
        batched_outputs = torch._functorch.predispatch._remove_batch_dim(
            result_2, 4, 20, 0
        )
        result_2 = None
        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs, 3, 20, 0
        )
        batched_outputs = None
        _vmap_decrement_nesting_1 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_1 = None
        batched_outputs_2 = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs_1, 2, 1, 0
        )
        batched_outputs_1 = None
        _vmap_decrement_nesting_2 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_2 = None
        causal_mask = torch._functorch.predispatch._remove_batch_dim(
            batched_outputs_2, 1, 1, 0
        )
        batched_outputs_2 = None
        _vmap_decrement_nesting_3 = (
            torch._functorch.predispatch._vmap_decrement_nesting()
        )
        _vmap_decrement_nesting_3 = None
        getitem = l_self_modules_model_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_model_modules_rotary_emb_buffers_inv_freq_ = None
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
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        hidden_states = inputs_embeds.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-06
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        hidden_states = rsqrt = None
        to_7 = hidden_states_1.to(torch.bfloat16)
        hidden_states_1 = None
        hidden_states_2 = (
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
            * to_7
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            to_7
        ) = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 20, -1, 128))
        linear = None
        hidden_states_3 = view.to(torch.float32)
        view = None
        pow_2 = hidden_states_3.pow(2)
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_1 = variance_1 + 1e-06
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_1)
        add_1 = None
        hidden_states_4 = hidden_states_3 * rsqrt_1
        hidden_states_3 = rsqrt_1 = None
        to_9 = hidden_states_4.to(torch.bfloat16)
        hidden_states_4 = None
        mul_5 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
            * to_9
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_9
        ) = None
        query_states = mul_5.transpose(1, 2)
        mul_5 = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 20, -1, 128))
        linear_1 = None
        hidden_states_5 = view_1.to(torch.float32)
        view_1 = None
        pow_3 = hidden_states_5.pow(2)
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_2 = variance_2 + 1e-06
        variance_2 = None
        rsqrt_2 = torch.rsqrt(add_2)
        add_2 = None
        hidden_states_6 = hidden_states_5 * rsqrt_2
        hidden_states_5 = rsqrt_2 = None
        to_11 = hidden_states_6.to(torch.bfloat16)
        hidden_states_6 = None
        mul_7 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
            * to_11
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_11
        ) = None
        key_states = mul_7.transpose(1, 2)
        mul_7 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = linear_2.view((1, 20, -1, 128))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_8 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 64, None))]
        x2 = query_states[(Ellipsis, slice(64, None, None))]
        query_states = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_9 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_8 + mul_9
        mul_8 = mul_9 = None
        mul_10 = key_states * cos_3
        cos_3 = None
        x1_1 = key_states[(Ellipsis, slice(None, 64, None))]
        x2_1 = key_states[(Ellipsis, slice(64, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_11 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_10 + mul_11
        mul_10 = mul_11 = None
        getitem_6 = k_embed[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed = None
        hidden_states_7 = getitem_6.expand(1, 8, 2, 20, 128)
        getitem_6 = None
        key = hidden_states_7.reshape(1, 16, 20, 128)
        hidden_states_7 = None
        getitem_7 = value_states[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states = None
        hidden_states_8 = getitem_7.expand(1, 8, 2, 20, 128)
        getitem_7 = None
        value = hidden_states_8.reshape(1, 16, 20, 128)
        hidden_states_8 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output = torch._C._nn.scaled_dot_product_attention(
            q_embed,
            key,
            value,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed = key = value = attention_mask_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        reshape_2 = attn_output_1.reshape(1, 20, -1)
        attn_output_1 = None
        attn_output_2 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_9 = inputs_embeds + attn_output_3
        inputs_embeds = attn_output_3 = None
        hidden_states_10 = hidden_states_9.to(torch.float32)
        pow_4 = hidden_states_10.pow(2)
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_6 = variance_3 + 1e-06
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_11 = hidden_states_10 * rsqrt_3
        hidden_states_10 = rsqrt_3 = None
        to_13 = hidden_states_11.to(torch.bfloat16)
        hidden_states_11 = None
        hidden_states_12 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_13
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_14 = silu * linear_5
        silu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_14,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_14 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_13 = hidden_states_9 + down_proj
        hidden_states_9 = down_proj = None
        hidden_states_14 = hidden_states_13.to(torch.float32)
        pow_5 = hidden_states_14.pow(2)
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_8 = variance_4 + 1e-06
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_8)
        add_8 = None
        hidden_states_15 = hidden_states_14 * rsqrt_4
        hidden_states_14 = rsqrt_4 = None
        to_15 = hidden_states_15.to(torch.bfloat16)
        hidden_states_15 = None
        hidden_states_16 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 20, -1, 128))
        linear_7 = None
        hidden_states_17 = view_3.to(torch.float32)
        view_3 = None
        pow_6 = hidden_states_17.pow(2)
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_9 = variance_5 + 1e-06
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_9)
        add_9 = None
        hidden_states_18 = hidden_states_17 * rsqrt_5
        hidden_states_17 = rsqrt_5 = None
        to_17 = hidden_states_18.to(torch.bfloat16)
        hidden_states_18 = None
        mul_18 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_
            * to_17
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_17
        ) = None
        query_states_1 = mul_18.transpose(1, 2)
        mul_18 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 20, -1, 128))
        linear_8 = None
        hidden_states_19 = view_4.to(torch.float32)
        view_4 = None
        pow_7 = hidden_states_19.pow(2)
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_10 = variance_6 + 1e-06
        variance_6 = None
        rsqrt_6 = torch.rsqrt(add_10)
        add_10 = None
        hidden_states_20 = hidden_states_19 * rsqrt_6
        hidden_states_19 = rsqrt_6 = None
        to_19 = hidden_states_20.to(torch.bfloat16)
        hidden_states_20 = None
        mul_20 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_
            * to_19
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_19
        ) = None
        key_states_1 = mul_20.transpose(1, 2)
        mul_20 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_16 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_9.view((1, 20, -1, 128))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_21 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 64, None))]
        x2_2 = query_states_1[(Ellipsis, slice(64, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_22 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_21 + mul_22
        mul_21 = mul_22 = None
        mul_23 = key_states_1 * cos_4
        cos_4 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 64, None))]
        x2_3 = key_states_1[(Ellipsis, slice(64, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_24 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_23 + mul_24
        mul_23 = mul_24 = None
        getitem_13 = k_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_1 = None
        hidden_states_21 = getitem_13.expand(1, 8, 2, 20, 128)
        getitem_13 = None
        key_1 = hidden_states_21.reshape(1, 16, 20, 128)
        hidden_states_21 = None
        getitem_14 = value_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_1 = None
        hidden_states_22 = getitem_14.expand(1, 8, 2, 20, 128)
        getitem_14 = None
        value_1 = hidden_states_22.reshape(1, 16, 20, 128)
        hidden_states_22 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            q_embed_1,
            key_1,
            value_1,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_1 = key_1 = value_1 = attention_mask_2 = None
        transpose_8 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_8.contiguous()
        transpose_8 = None
        reshape_5 = attn_output_5.reshape(1, 20, -1)
        attn_output_5 = None
        attn_output_6 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_23 = hidden_states_13 + attn_output_7
        hidden_states_13 = attn_output_7 = None
        hidden_states_24 = hidden_states_23.to(torch.float32)
        pow_8 = hidden_states_24.pow(2)
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_14 = variance_7 + 1e-06
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_14)
        add_14 = None
        hidden_states_25 = hidden_states_24 * rsqrt_7
        hidden_states_24 = rsqrt_7 = None
        to_21 = hidden_states_25.to(torch.bfloat16)
        hidden_states_25 = None
        hidden_states_26 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_21
        ) = None
        linear_11 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_26,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_26 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_27 = silu_1 * linear_12
        silu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_27,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_27 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_27 = hidden_states_23 + down_proj_1
        hidden_states_23 = down_proj_1 = None
        hidden_states_28 = hidden_states_27.to(torch.float32)
        pow_9 = hidden_states_28.pow(2)
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_16 = variance_8 + 1e-06
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_29 = hidden_states_28 * rsqrt_8
        hidden_states_28 = rsqrt_8 = None
        to_23 = hidden_states_29.to(torch.bfloat16)
        hidden_states_29 = None
        hidden_states_30 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_23
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 20, -1, 128))
        linear_14 = None
        hidden_states_31 = view_6.to(torch.float32)
        view_6 = None
        pow_10 = hidden_states_31.pow(2)
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_17 = variance_9 + 1e-06
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_17)
        add_17 = None
        hidden_states_32 = hidden_states_31 * rsqrt_9
        hidden_states_31 = rsqrt_9 = None
        to_25 = hidden_states_32.to(torch.bfloat16)
        hidden_states_32 = None
        mul_31 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_
            * to_25
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_25
        ) = None
        query_states_2 = mul_31.transpose(1, 2)
        mul_31 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 20, -1, 128))
        linear_15 = None
        hidden_states_33 = view_7.to(torch.float32)
        view_7 = None
        pow_11 = hidden_states_33.pow(2)
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_18 = variance_10 + 1e-06
        variance_10 = None
        rsqrt_10 = torch.rsqrt(add_18)
        add_18 = None
        hidden_states_34 = hidden_states_33 * rsqrt_10
        hidden_states_33 = rsqrt_10 = None
        to_27 = hidden_states_34.to(torch.bfloat16)
        hidden_states_34 = None
        mul_33 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_
            * to_27
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_27
        ) = None
        key_states_2 = mul_33.transpose(1, 2)
        mul_33 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_16.view((1, 20, -1, 128))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_34 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 64, None))]
        x2_4 = query_states_2[(Ellipsis, slice(64, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_35 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_34 + mul_35
        mul_34 = mul_35 = None
        mul_36 = key_states_2 * cos_5
        cos_5 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 64, None))]
        x2_5 = key_states_2[(Ellipsis, slice(64, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_37 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_36 + mul_37
        mul_36 = mul_37 = None
        getitem_20 = k_embed_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_2 = None
        hidden_states_35 = getitem_20.expand(1, 8, 2, 20, 128)
        getitem_20 = None
        key_2 = hidden_states_35.reshape(1, 16, 20, 128)
        hidden_states_35 = None
        getitem_21 = value_states_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_2 = None
        hidden_states_36 = getitem_21.expand(1, 8, 2, 20, 128)
        getitem_21 = None
        value_2 = hidden_states_36.reshape(1, 16, 20, 128)
        hidden_states_36 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            q_embed_2,
            key_2,
            value_2,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_2 = key_2 = value_2 = attention_mask_3 = None
        transpose_12 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_12.contiguous()
        transpose_12 = None
        reshape_8 = attn_output_9.reshape(1, 20, -1)
        attn_output_9 = None
        attn_output_10 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_37 = hidden_states_27 + attn_output_11
        hidden_states_27 = attn_output_11 = None
        hidden_states_38 = hidden_states_37.to(torch.float32)
        pow_12 = hidden_states_38.pow(2)
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_22 = variance_11 + 1e-06
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_39 = hidden_states_38 * rsqrt_11
        hidden_states_38 = rsqrt_11 = None
        to_29 = hidden_states_39.to(torch.bfloat16)
        hidden_states_39 = None
        hidden_states_40 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_29
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_40 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_40 = silu_2 * linear_19
        silu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_40,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_40 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_41 = hidden_states_37 + down_proj_2
        hidden_states_37 = down_proj_2 = None
        hidden_states_42 = hidden_states_41.to(torch.float32)
        pow_13 = hidden_states_42.pow(2)
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_24 = variance_12 + 1e-06
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_43 = hidden_states_42 * rsqrt_12
        hidden_states_42 = rsqrt_12 = None
        to_31 = hidden_states_43.to(torch.bfloat16)
        hidden_states_43 = None
        hidden_states_44 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_31
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 20, -1, 128))
        linear_21 = None
        hidden_states_45 = view_9.to(torch.float32)
        view_9 = None
        pow_14 = hidden_states_45.pow(2)
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_25 = variance_13 + 1e-06
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_25)
        add_25 = None
        hidden_states_46 = hidden_states_45 * rsqrt_13
        hidden_states_45 = rsqrt_13 = None
        to_33 = hidden_states_46.to(torch.bfloat16)
        hidden_states_46 = None
        mul_44 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_
            * to_33
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_33
        ) = None
        query_states_3 = mul_44.transpose(1, 2)
        mul_44 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 20, -1, 128))
        linear_22 = None
        hidden_states_47 = view_10.to(torch.float32)
        view_10 = None
        pow_15 = hidden_states_47.pow(2)
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_26 = variance_14 + 1e-06
        variance_14 = None
        rsqrt_14 = torch.rsqrt(add_26)
        add_26 = None
        hidden_states_48 = hidden_states_47 * rsqrt_14
        hidden_states_47 = rsqrt_14 = None
        to_35 = hidden_states_48.to(torch.bfloat16)
        hidden_states_48 = None
        mul_46 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_
            * to_35
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_35
        ) = None
        key_states_3 = mul_46.transpose(1, 2)
        mul_46 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_44,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_44 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_23.view((1, 20, -1, 128))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_47 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 64, None))]
        x2_6 = query_states_3[(Ellipsis, slice(64, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_48 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_47 + mul_48
        mul_47 = mul_48 = None
        mul_49 = key_states_3 * cos_6
        cos_6 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 64, None))]
        x2_7 = key_states_3[(Ellipsis, slice(64, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_50 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_49 + mul_50
        mul_49 = mul_50 = None
        getitem_27 = k_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_3 = None
        hidden_states_49 = getitem_27.expand(1, 8, 2, 20, 128)
        getitem_27 = None
        key_3 = hidden_states_49.reshape(1, 16, 20, 128)
        hidden_states_49 = None
        getitem_28 = value_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_3 = None
        hidden_states_50 = getitem_28.expand(1, 8, 2, 20, 128)
        getitem_28 = None
        value_3 = hidden_states_50.reshape(1, 16, 20, 128)
        hidden_states_50 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            q_embed_3,
            key_3,
            value_3,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_3 = key_3 = value_3 = attention_mask_4 = None
        transpose_16 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_16.contiguous()
        transpose_16 = None
        reshape_11 = attn_output_13.reshape(1, 20, -1)
        attn_output_13 = None
        attn_output_14 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_51 = hidden_states_41 + attn_output_15
        hidden_states_41 = attn_output_15 = None
        hidden_states_52 = hidden_states_51.to(torch.float32)
        pow_16 = hidden_states_52.pow(2)
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_30 = variance_15 + 1e-06
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_53 = hidden_states_52 * rsqrt_15
        hidden_states_52 = rsqrt_15 = None
        to_37 = hidden_states_53.to(torch.bfloat16)
        hidden_states_53 = None
        hidden_states_54 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_37
        ) = None
        linear_25 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_54,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_54 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_53 = silu_3 * linear_26
        silu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_53,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_53 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_55 = hidden_states_51 + down_proj_3
        hidden_states_51 = down_proj_3 = None
        hidden_states_56 = hidden_states_55.to(torch.float32)
        pow_17 = hidden_states_56.pow(2)
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_32 = variance_16 + 1e-06
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_32)
        add_32 = None
        hidden_states_57 = hidden_states_56 * rsqrt_16
        hidden_states_56 = rsqrt_16 = None
        to_39 = hidden_states_57.to(torch.bfloat16)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_39
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 20, -1, 128))
        linear_28 = None
        hidden_states_59 = view_12.to(torch.float32)
        view_12 = None
        pow_18 = hidden_states_59.pow(2)
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_33 = variance_17 + 1e-06
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_33)
        add_33 = None
        hidden_states_60 = hidden_states_59 * rsqrt_17
        hidden_states_59 = rsqrt_17 = None
        to_41 = hidden_states_60.to(torch.bfloat16)
        hidden_states_60 = None
        mul_57 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_
            * to_41
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_41
        ) = None
        query_states_4 = mul_57.transpose(1, 2)
        mul_57 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 20, -1, 128))
        linear_29 = None
        hidden_states_61 = view_13.to(torch.float32)
        view_13 = None
        pow_19 = hidden_states_61.pow(2)
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_34 = variance_18 + 1e-06
        variance_18 = None
        rsqrt_18 = torch.rsqrt(add_34)
        add_34 = None
        hidden_states_62 = hidden_states_61 * rsqrt_18
        hidden_states_61 = rsqrt_18 = None
        to_43 = hidden_states_62.to(torch.bfloat16)
        hidden_states_62 = None
        mul_59 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_
            * to_43
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_43
        ) = None
        key_states_4 = mul_59.transpose(1, 2)
        mul_59 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_58 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_30.view((1, 20, -1, 128))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_60 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 64, None))]
        x2_8 = query_states_4[(Ellipsis, slice(64, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_61 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_60 + mul_61
        mul_60 = mul_61 = None
        mul_62 = key_states_4 * cos_7
        cos_7 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 64, None))]
        x2_9 = key_states_4[(Ellipsis, slice(64, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_63 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_62 + mul_63
        mul_62 = mul_63 = None
        getitem_34 = k_embed_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_4 = None
        hidden_states_63 = getitem_34.expand(1, 8, 2, 20, 128)
        getitem_34 = None
        key_4 = hidden_states_63.reshape(1, 16, 20, 128)
        hidden_states_63 = None
        getitem_35 = value_states_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_4 = None
        hidden_states_64 = getitem_35.expand(1, 8, 2, 20, 128)
        getitem_35 = None
        value_4 = hidden_states_64.reshape(1, 16, 20, 128)
        hidden_states_64 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            q_embed_4,
            key_4,
            value_4,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_4 = key_4 = value_4 = attention_mask_5 = None
        transpose_20 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_20.contiguous()
        transpose_20 = None
        reshape_14 = attn_output_17.reshape(1, 20, -1)
        attn_output_17 = None
        attn_output_18 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_65 = hidden_states_55 + attn_output_19
        hidden_states_55 = attn_output_19 = None
        hidden_states_66 = hidden_states_65.to(torch.float32)
        pow_20 = hidden_states_66.pow(2)
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_38 = variance_19 + 1e-06
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_38)
        add_38 = None
        hidden_states_67 = hidden_states_66 * rsqrt_19
        hidden_states_66 = rsqrt_19 = None
        to_45 = hidden_states_67.to(torch.bfloat16)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_45
        ) = None
        linear_32 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_68 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_66 = silu_4 * linear_33
        silu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_66,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_66 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_69 = hidden_states_65 + down_proj_4
        hidden_states_65 = down_proj_4 = None
        hidden_states_70 = hidden_states_69.to(torch.float32)
        pow_21 = hidden_states_70.pow(2)
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_40 = variance_20 + 1e-06
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_71 = hidden_states_70 * rsqrt_20
        hidden_states_70 = rsqrt_20 = None
        to_47 = hidden_states_71.to(torch.bfloat16)
        hidden_states_71 = None
        hidden_states_72 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_47
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_47
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 20, -1, 128))
        linear_35 = None
        hidden_states_73 = view_15.to(torch.float32)
        view_15 = None
        pow_22 = hidden_states_73.pow(2)
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_41 = variance_21 + 1e-06
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_41)
        add_41 = None
        hidden_states_74 = hidden_states_73 * rsqrt_21
        hidden_states_73 = rsqrt_21 = None
        to_49 = hidden_states_74.to(torch.bfloat16)
        hidden_states_74 = None
        mul_70 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_
            * to_49
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_49
        ) = None
        query_states_5 = mul_70.transpose(1, 2)
        mul_70 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 20, -1, 128))
        linear_36 = None
        hidden_states_75 = view_16.to(torch.float32)
        view_16 = None
        pow_23 = hidden_states_75.pow(2)
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_42 = variance_22 + 1e-06
        variance_22 = None
        rsqrt_22 = torch.rsqrt(add_42)
        add_42 = None
        hidden_states_76 = hidden_states_75 * rsqrt_22
        hidden_states_75 = rsqrt_22 = None
        to_51 = hidden_states_76.to(torch.bfloat16)
        hidden_states_76 = None
        mul_72 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_
            * to_51
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_51
        ) = None
        key_states_5 = mul_72.transpose(1, 2)
        mul_72 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_37.view((1, 20, -1, 128))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_73 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 64, None))]
        x2_10 = query_states_5[(Ellipsis, slice(64, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_74 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_73 + mul_74
        mul_73 = mul_74 = None
        mul_75 = key_states_5 * cos_8
        cos_8 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 64, None))]
        x2_11 = key_states_5[(Ellipsis, slice(64, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_76 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_75 + mul_76
        mul_75 = mul_76 = None
        getitem_41 = k_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_5 = None
        hidden_states_77 = getitem_41.expand(1, 8, 2, 20, 128)
        getitem_41 = None
        key_5 = hidden_states_77.reshape(1, 16, 20, 128)
        hidden_states_77 = None
        getitem_42 = value_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_5 = None
        hidden_states_78 = getitem_42.expand(1, 8, 2, 20, 128)
        getitem_42 = None
        value_5 = hidden_states_78.reshape(1, 16, 20, 128)
        hidden_states_78 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            q_embed_5,
            key_5,
            value_5,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_5 = key_5 = value_5 = attention_mask_6 = None
        transpose_24 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_24.contiguous()
        transpose_24 = None
        reshape_17 = attn_output_21.reshape(1, 20, -1)
        attn_output_21 = None
        attn_output_22 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_79 = hidden_states_69 + attn_output_23
        hidden_states_69 = attn_output_23 = None
        hidden_states_80 = hidden_states_79.to(torch.float32)
        pow_24 = hidden_states_80.pow(2)
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_46 = variance_23 + 1e-06
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_81 = hidden_states_80 * rsqrt_23
        hidden_states_80 = rsqrt_23 = None
        to_53 = hidden_states_81.to(torch.bfloat16)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_53
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_53
        ) = None
        linear_39 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_82 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_79 = silu_5 * linear_40
        silu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_79,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_79 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_83 = hidden_states_79 + down_proj_5
        hidden_states_79 = down_proj_5 = None
        hidden_states_84 = hidden_states_83.to(torch.float32)
        pow_25 = hidden_states_84.pow(2)
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_48 = variance_24 + 1e-06
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_85 = hidden_states_84 * rsqrt_24
        hidden_states_84 = rsqrt_24 = None
        to_55 = hidden_states_85.to(torch.bfloat16)
        hidden_states_85 = None
        hidden_states_86 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_55
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_55
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 20, -1, 128))
        linear_42 = None
        hidden_states_87 = view_18.to(torch.float32)
        view_18 = None
        pow_26 = hidden_states_87.pow(2)
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_49 = variance_25 + 1e-06
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_49)
        add_49 = None
        hidden_states_88 = hidden_states_87 * rsqrt_25
        hidden_states_87 = rsqrt_25 = None
        to_57 = hidden_states_88.to(torch.bfloat16)
        hidden_states_88 = None
        mul_83 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_
            * to_57
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_57
        ) = None
        query_states_6 = mul_83.transpose(1, 2)
        mul_83 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 20, -1, 128))
        linear_43 = None
        hidden_states_89 = view_19.to(torch.float32)
        view_19 = None
        pow_27 = hidden_states_89.pow(2)
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_50 = variance_26 + 1e-06
        variance_26 = None
        rsqrt_26 = torch.rsqrt(add_50)
        add_50 = None
        hidden_states_90 = hidden_states_89 * rsqrt_26
        hidden_states_89 = rsqrt_26 = None
        to_59 = hidden_states_90.to(torch.bfloat16)
        hidden_states_90 = None
        mul_85 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_
            * to_59
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_59
        ) = None
        key_states_6 = mul_85.transpose(1, 2)
        mul_85 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_86,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_86 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_44.view((1, 20, -1, 128))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_86 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 64, None))]
        x2_12 = query_states_6[(Ellipsis, slice(64, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_87 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_86 + mul_87
        mul_86 = mul_87 = None
        mul_88 = key_states_6 * cos_9
        cos_9 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 64, None))]
        x2_13 = key_states_6[(Ellipsis, slice(64, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_89 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_88 + mul_89
        mul_88 = mul_89 = None
        getitem_48 = k_embed_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_6 = None
        hidden_states_91 = getitem_48.expand(1, 8, 2, 20, 128)
        getitem_48 = None
        key_6 = hidden_states_91.reshape(1, 16, 20, 128)
        hidden_states_91 = None
        getitem_49 = value_states_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_6 = None
        hidden_states_92 = getitem_49.expand(1, 8, 2, 20, 128)
        getitem_49 = None
        value_6 = hidden_states_92.reshape(1, 16, 20, 128)
        hidden_states_92 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            q_embed_6,
            key_6,
            value_6,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_6 = key_6 = value_6 = attention_mask_7 = None
        transpose_28 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_28.contiguous()
        transpose_28 = None
        reshape_20 = attn_output_25.reshape(1, 20, -1)
        attn_output_25 = None
        attn_output_26 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_93 = hidden_states_83 + attn_output_27
        hidden_states_83 = attn_output_27 = None
        hidden_states_94 = hidden_states_93.to(torch.float32)
        pow_28 = hidden_states_94.pow(2)
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_54 = variance_27 + 1e-06
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_95 = hidden_states_94 * rsqrt_27
        hidden_states_94 = rsqrt_27 = None
        to_61 = hidden_states_95.to(torch.bfloat16)
        hidden_states_95 = None
        hidden_states_96 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_61
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_61
        ) = None
        linear_46 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_96 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_92 = silu_6 * linear_47
        silu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_92,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_92 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_97 = hidden_states_93 + down_proj_6
        hidden_states_93 = down_proj_6 = None
        hidden_states_98 = hidden_states_97.to(torch.float32)
        pow_29 = hidden_states_98.pow(2)
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_56 = variance_28 + 1e-06
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_56)
        add_56 = None
        hidden_states_99 = hidden_states_98 * rsqrt_28
        hidden_states_98 = rsqrt_28 = None
        to_63 = hidden_states_99.to(torch.bfloat16)
        hidden_states_99 = None
        hidden_states_100 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_63
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_63
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 20, -1, 128))
        linear_49 = None
        hidden_states_101 = view_21.to(torch.float32)
        view_21 = None
        pow_30 = hidden_states_101.pow(2)
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_57 = variance_29 + 1e-06
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_57)
        add_57 = None
        hidden_states_102 = hidden_states_101 * rsqrt_29
        hidden_states_101 = rsqrt_29 = None
        to_65 = hidden_states_102.to(torch.bfloat16)
        hidden_states_102 = None
        mul_96 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_
            * to_65
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_65
        ) = None
        query_states_7 = mul_96.transpose(1, 2)
        mul_96 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 20, -1, 128))
        linear_50 = None
        hidden_states_103 = view_22.to(torch.float32)
        view_22 = None
        pow_31 = hidden_states_103.pow(2)
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_58 = variance_30 + 1e-06
        variance_30 = None
        rsqrt_30 = torch.rsqrt(add_58)
        add_58 = None
        hidden_states_104 = hidden_states_103 * rsqrt_30
        hidden_states_103 = rsqrt_30 = None
        to_67 = hidden_states_104.to(torch.bfloat16)
        hidden_states_104 = None
        mul_98 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_
            * to_67
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_67
        ) = None
        key_states_7 = mul_98.transpose(1, 2)
        mul_98 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_100 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_51.view((1, 20, -1, 128))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_99 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 64, None))]
        x2_14 = query_states_7[(Ellipsis, slice(64, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_100 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_99 + mul_100
        mul_99 = mul_100 = None
        mul_101 = key_states_7 * cos_10
        cos_10 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 64, None))]
        x2_15 = key_states_7[(Ellipsis, slice(64, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_102 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_101 + mul_102
        mul_101 = mul_102 = None
        getitem_55 = k_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_7 = None
        hidden_states_105 = getitem_55.expand(1, 8, 2, 20, 128)
        getitem_55 = None
        key_7 = hidden_states_105.reshape(1, 16, 20, 128)
        hidden_states_105 = None
        getitem_56 = value_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_7 = None
        hidden_states_106 = getitem_56.expand(1, 8, 2, 20, 128)
        getitem_56 = None
        value_7 = hidden_states_106.reshape(1, 16, 20, 128)
        hidden_states_106 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            q_embed_7,
            key_7,
            value_7,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_7 = key_7 = value_7 = attention_mask_8 = None
        transpose_32 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_32.contiguous()
        transpose_32 = None
        reshape_23 = attn_output_29.reshape(1, 20, -1)
        attn_output_29 = None
        attn_output_30 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_107 = hidden_states_97 + attn_output_31
        hidden_states_97 = attn_output_31 = None
        hidden_states_108 = hidden_states_107.to(torch.float32)
        pow_32 = hidden_states_108.pow(2)
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_62 = variance_31 + 1e-06
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_62)
        add_62 = None
        hidden_states_109 = hidden_states_108 * rsqrt_31
        hidden_states_108 = rsqrt_31 = None
        to_69 = hidden_states_109.to(torch.bfloat16)
        hidden_states_109 = None
        hidden_states_110 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_69
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_69
        ) = None
        linear_53 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_110 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_105 = silu_7 * linear_54
        silu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_105,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_105 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_111 = hidden_states_107 + down_proj_7
        hidden_states_107 = down_proj_7 = None
        hidden_states_112 = hidden_states_111.to(torch.float32)
        pow_33 = hidden_states_112.pow(2)
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_64 = variance_32 + 1e-06
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_113 = hidden_states_112 * rsqrt_32
        hidden_states_112 = rsqrt_32 = None
        to_71 = hidden_states_113.to(torch.bfloat16)
        hidden_states_113 = None
        hidden_states_114 = (
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_71
        )
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_71
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_56.view((1, 20, -1, 128))
        linear_56 = None
        hidden_states_115 = view_24.to(torch.float32)
        view_24 = None
        pow_34 = hidden_states_115.pow(2)
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_65 = variance_33 + 1e-06
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_65)
        add_65 = None
        hidden_states_116 = hidden_states_115 * rsqrt_33
        hidden_states_115 = rsqrt_33 = None
        to_73 = hidden_states_116.to(torch.bfloat16)
        hidden_states_116 = None
        mul_109 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_
            * to_73
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_73
        ) = None
        query_states_8 = mul_109.transpose(1, 2)
        mul_109 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_57.view((1, 20, -1, 128))
        linear_57 = None
        hidden_states_117 = view_25.to(torch.float32)
        view_25 = None
        pow_35 = hidden_states_117.pow(2)
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_66 = variance_34 + 1e-06
        variance_34 = None
        rsqrt_34 = torch.rsqrt(add_66)
        add_66 = None
        hidden_states_118 = hidden_states_117 * rsqrt_34
        hidden_states_117 = rsqrt_34 = None
        to_75 = hidden_states_118.to(torch.bfloat16)
        hidden_states_118 = None
        mul_111 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_
            * to_75
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_75
        ) = None
        key_states_8 = mul_111.transpose(1, 2)
        mul_111 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_114,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_114 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = linear_58.view((1, 20, -1, 128))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_112 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 64, None))]
        x2_16 = query_states_8[(Ellipsis, slice(64, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_113 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_112 + mul_113
        mul_112 = mul_113 = None
        mul_114 = key_states_8 * cos_11
        cos_11 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 64, None))]
        x2_17 = key_states_8[(Ellipsis, slice(64, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_115 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_114 + mul_115
        mul_114 = mul_115 = None
        getitem_62 = k_embed_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_8 = None
        hidden_states_119 = getitem_62.expand(1, 8, 2, 20, 128)
        getitem_62 = None
        key_8 = hidden_states_119.reshape(1, 16, 20, 128)
        hidden_states_119 = None
        getitem_63 = value_states_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_8 = None
        hidden_states_120 = getitem_63.expand(1, 8, 2, 20, 128)
        getitem_63 = None
        value_8 = hidden_states_120.reshape(1, 16, 20, 128)
        hidden_states_120 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            q_embed_8,
            key_8,
            value_8,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_8 = key_8 = value_8 = attention_mask_9 = None
        transpose_36 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_36.contiguous()
        transpose_36 = None
        reshape_26 = attn_output_33.reshape(1, 20, -1)
        attn_output_33 = None
        attn_output_34 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_121 = hidden_states_111 + attn_output_35
        hidden_states_111 = attn_output_35 = None
        hidden_states_122 = hidden_states_121.to(torch.float32)
        pow_36 = hidden_states_122.pow(2)
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_70 = variance_35 + 1e-06
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_123 = hidden_states_122 * rsqrt_35
        hidden_states_122 = rsqrt_35 = None
        to_77 = hidden_states_123.to(torch.bfloat16)
        hidden_states_123 = None
        hidden_states_124 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_77
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_77
        ) = None
        linear_60 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_60, inplace=False)
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_124,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_124 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_118 = silu_8 * linear_61
        silu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_118,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_118 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_125 = hidden_states_121 + down_proj_8
        hidden_states_121 = down_proj_8 = None
        hidden_states_126 = hidden_states_125.to(torch.float32)
        pow_37 = hidden_states_126.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_72 = variance_36 + 1e-06
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_127 = hidden_states_126 * rsqrt_36
        hidden_states_126 = rsqrt_36 = None
        to_79 = hidden_states_127.to(torch.bfloat16)
        hidden_states_127 = None
        hidden_states_128 = (
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_79
        )
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_79
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_63.view((1, 20, -1, 128))
        linear_63 = None
        hidden_states_129 = view_27.to(torch.float32)
        view_27 = None
        pow_38 = hidden_states_129.pow(2)
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_73 = variance_37 + 1e-06
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_73)
        add_73 = None
        hidden_states_130 = hidden_states_129 * rsqrt_37
        hidden_states_129 = rsqrt_37 = None
        to_81 = hidden_states_130.to(torch.bfloat16)
        hidden_states_130 = None
        mul_122 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_
            * to_81
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_81
        ) = None
        query_states_9 = mul_122.transpose(1, 2)
        mul_122 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_64.view((1, 20, -1, 128))
        linear_64 = None
        hidden_states_131 = view_28.to(torch.float32)
        view_28 = None
        pow_39 = hidden_states_131.pow(2)
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_74 = variance_38 + 1e-06
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_74)
        add_74 = None
        hidden_states_132 = hidden_states_131 * rsqrt_38
        hidden_states_131 = rsqrt_38 = None
        to_83 = hidden_states_132.to(torch.bfloat16)
        hidden_states_132 = None
        mul_124 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_
            * to_83
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_83
        ) = None
        key_states_9 = mul_124.transpose(1, 2)
        mul_124 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_128 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = linear_65.view((1, 20, -1, 128))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_125 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 64, None))]
        x2_18 = query_states_9[(Ellipsis, slice(64, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_126 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_125 + mul_126
        mul_125 = mul_126 = None
        mul_127 = key_states_9 * cos_12
        cos_12 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 64, None))]
        x2_19 = key_states_9[(Ellipsis, slice(64, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_128 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_127 + mul_128
        mul_127 = mul_128 = None
        getitem_69 = k_embed_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_9 = None
        hidden_states_133 = getitem_69.expand(1, 8, 2, 20, 128)
        getitem_69 = None
        key_9 = hidden_states_133.reshape(1, 16, 20, 128)
        hidden_states_133 = None
        getitem_70 = value_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_9 = None
        hidden_states_134 = getitem_70.expand(1, 8, 2, 20, 128)
        getitem_70 = None
        value_9 = hidden_states_134.reshape(1, 16, 20, 128)
        hidden_states_134 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            q_embed_9,
            key_9,
            value_9,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_9 = key_9 = value_9 = attention_mask_10 = None
        transpose_40 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_40.contiguous()
        transpose_40 = None
        reshape_29 = attn_output_37.reshape(1, 20, -1)
        attn_output_37 = None
        attn_output_38 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_135 = hidden_states_125 + attn_output_39
        hidden_states_125 = attn_output_39 = None
        hidden_states_136 = hidden_states_135.to(torch.float32)
        pow_40 = hidden_states_136.pow(2)
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_78 = variance_39 + 1e-06
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_137 = hidden_states_136 * rsqrt_39
        hidden_states_136 = rsqrt_39 = None
        to_85 = hidden_states_137.to(torch.bfloat16)
        hidden_states_137 = None
        hidden_states_138 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_85
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_85
        ) = None
        linear_67 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_67, inplace=False)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_138 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_131 = silu_9 * linear_68
        silu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_131,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_131 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_139 = hidden_states_135 + down_proj_9
        hidden_states_135 = down_proj_9 = None
        hidden_states_140 = hidden_states_139.to(torch.float32)
        pow_41 = hidden_states_140.pow(2)
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_80 = variance_40 + 1e-06
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_80)
        add_80 = None
        hidden_states_141 = hidden_states_140 * rsqrt_40
        hidden_states_140 = rsqrt_40 = None
        to_87 = hidden_states_141.to(torch.bfloat16)
        hidden_states_141 = None
        hidden_states_142 = (
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_87
        )
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_87
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_70.view((1, 20, -1, 128))
        linear_70 = None
        hidden_states_143 = view_30.to(torch.float32)
        view_30 = None
        pow_42 = hidden_states_143.pow(2)
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_81 = variance_41 + 1e-06
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_81)
        add_81 = None
        hidden_states_144 = hidden_states_143 * rsqrt_41
        hidden_states_143 = rsqrt_41 = None
        to_89 = hidden_states_144.to(torch.bfloat16)
        hidden_states_144 = None
        mul_135 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_
            * to_89
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_89
        ) = None
        query_states_10 = mul_135.transpose(1, 2)
        mul_135 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_71.view((1, 20, -1, 128))
        linear_71 = None
        hidden_states_145 = view_31.to(torch.float32)
        view_31 = None
        pow_43 = hidden_states_145.pow(2)
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_82 = variance_42 + 1e-06
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_82)
        add_82 = None
        hidden_states_146 = hidden_states_145 * rsqrt_42
        hidden_states_145 = rsqrt_42 = None
        to_91 = hidden_states_146.to(torch.bfloat16)
        hidden_states_146 = None
        mul_137 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_
            * to_91
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_91
        ) = None
        key_states_10 = mul_137.transpose(1, 2)
        mul_137 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_142 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = linear_72.view((1, 20, -1, 128))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_138 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 64, None))]
        x2_20 = query_states_10[(Ellipsis, slice(64, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_139 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_138 + mul_139
        mul_138 = mul_139 = None
        mul_140 = key_states_10 * cos_13
        cos_13 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 64, None))]
        x2_21 = key_states_10[(Ellipsis, slice(64, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_141 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_140 + mul_141
        mul_140 = mul_141 = None
        getitem_76 = k_embed_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_10 = None
        hidden_states_147 = getitem_76.expand(1, 8, 2, 20, 128)
        getitem_76 = None
        key_10 = hidden_states_147.reshape(1, 16, 20, 128)
        hidden_states_147 = None
        getitem_77 = value_states_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_10 = None
        hidden_states_148 = getitem_77.expand(1, 8, 2, 20, 128)
        getitem_77 = None
        value_10 = hidden_states_148.reshape(1, 16, 20, 128)
        hidden_states_148 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            q_embed_10,
            key_10,
            value_10,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_10 = key_10 = value_10 = attention_mask_11 = None
        transpose_44 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_44.contiguous()
        transpose_44 = None
        reshape_32 = attn_output_41.reshape(1, 20, -1)
        attn_output_41 = None
        attn_output_42 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_149 = hidden_states_139 + attn_output_43
        hidden_states_139 = attn_output_43 = None
        hidden_states_150 = hidden_states_149.to(torch.float32)
        pow_44 = hidden_states_150.pow(2)
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_86 = variance_43 + 1e-06
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_86)
        add_86 = None
        hidden_states_151 = hidden_states_150 * rsqrt_43
        hidden_states_150 = rsqrt_43 = None
        to_93 = hidden_states_151.to(torch.bfloat16)
        hidden_states_151 = None
        hidden_states_152 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_93
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_93
        ) = None
        linear_74 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_152 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_144 = silu_10 * linear_75
        silu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_144,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_144 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_153 = hidden_states_149 + down_proj_10
        hidden_states_149 = down_proj_10 = None
        hidden_states_154 = hidden_states_153.to(torch.float32)
        pow_45 = hidden_states_154.pow(2)
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_88 = variance_44 + 1e-06
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_155 = hidden_states_154 * rsqrt_44
        hidden_states_154 = rsqrt_44 = None
        to_95 = hidden_states_155.to(torch.bfloat16)
        hidden_states_155 = None
        hidden_states_156 = (
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_95
        )
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_95
        ) = None
        linear_77 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_77.view((1, 20, -1, 128))
        linear_77 = None
        hidden_states_157 = view_33.to(torch.float32)
        view_33 = None
        pow_46 = hidden_states_157.pow(2)
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_89 = variance_45 + 1e-06
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_89)
        add_89 = None
        hidden_states_158 = hidden_states_157 * rsqrt_45
        hidden_states_157 = rsqrt_45 = None
        to_97 = hidden_states_158.to(torch.bfloat16)
        hidden_states_158 = None
        mul_148 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_
            * to_97
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_97
        ) = None
        query_states_11 = mul_148.transpose(1, 2)
        mul_148 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_78.view((1, 20, -1, 128))
        linear_78 = None
        hidden_states_159 = view_34.to(torch.float32)
        view_34 = None
        pow_47 = hidden_states_159.pow(2)
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_90 = variance_46 + 1e-06
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_90)
        add_90 = None
        hidden_states_160 = hidden_states_159 * rsqrt_46
        hidden_states_159 = rsqrt_46 = None
        to_99 = hidden_states_160.to(torch.bfloat16)
        hidden_states_160 = None
        mul_150 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_
            * to_99
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_99
        ) = None
        key_states_11 = mul_150.transpose(1, 2)
        mul_150 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_156,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_156 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = linear_79.view((1, 20, -1, 128))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_151 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 64, None))]
        x2_22 = query_states_11[(Ellipsis, slice(64, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_152 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_151 + mul_152
        mul_151 = mul_152 = None
        mul_153 = key_states_11 * cos_14
        cos_14 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 64, None))]
        x2_23 = key_states_11[(Ellipsis, slice(64, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_154 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_153 + mul_154
        mul_153 = mul_154 = None
        getitem_83 = k_embed_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_11 = None
        hidden_states_161 = getitem_83.expand(1, 8, 2, 20, 128)
        getitem_83 = None
        key_11 = hidden_states_161.reshape(1, 16, 20, 128)
        hidden_states_161 = None
        getitem_84 = value_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_11 = None
        hidden_states_162 = getitem_84.expand(1, 8, 2, 20, 128)
        getitem_84 = None
        value_11 = hidden_states_162.reshape(1, 16, 20, 128)
        hidden_states_162 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            q_embed_11,
            key_11,
            value_11,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_11 = key_11 = value_11 = attention_mask_12 = None
        transpose_48 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_48.contiguous()
        transpose_48 = None
        reshape_35 = attn_output_45.reshape(1, 20, -1)
        attn_output_45 = None
        attn_output_46 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_163 = hidden_states_153 + attn_output_47
        hidden_states_153 = attn_output_47 = None
        hidden_states_164 = hidden_states_163.to(torch.float32)
        pow_48 = hidden_states_164.pow(2)
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_94 = variance_47 + 1e-06
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_165 = hidden_states_164 * rsqrt_47
        hidden_states_164 = rsqrt_47 = None
        to_101 = hidden_states_165.to(torch.bfloat16)
        hidden_states_165 = None
        hidden_states_166 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_101
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_101
        ) = None
        linear_81 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_81, inplace=False)
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_166,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_166 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_157 = silu_11 * linear_82
        silu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_157,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_157 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_167 = hidden_states_163 + down_proj_11
        hidden_states_163 = down_proj_11 = None
        hidden_states_168 = hidden_states_167.to(torch.float32)
        pow_49 = hidden_states_168.pow(2)
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_96 = variance_48 + 1e-06
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_169 = hidden_states_168 * rsqrt_48
        hidden_states_168 = rsqrt_48 = None
        to_103 = hidden_states_169.to(torch.bfloat16)
        hidden_states_169 = None
        hidden_states_170 = (
            l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
            * to_103
        )
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            to_103
        ) = None
        linear_84 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_36 = linear_84.view((1, 20, -1, 128))
        linear_84 = None
        hidden_states_171 = view_36.to(torch.float32)
        view_36 = None
        pow_50 = hidden_states_171.pow(2)
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_97 = variance_49 + 1e-06
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_97)
        add_97 = None
        hidden_states_172 = hidden_states_171 * rsqrt_49
        hidden_states_171 = rsqrt_49 = None
        to_105 = hidden_states_172.to(torch.bfloat16)
        hidden_states_172 = None
        mul_161 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_
            * to_105
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_105
        ) = None
        query_states_12 = mul_161.transpose(1, 2)
        mul_161 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_85.view((1, 20, -1, 128))
        linear_85 = None
        hidden_states_173 = view_37.to(torch.float32)
        view_37 = None
        pow_51 = hidden_states_173.pow(2)
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_98 = variance_50 + 1e-06
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_98)
        add_98 = None
        hidden_states_174 = hidden_states_173 * rsqrt_50
        hidden_states_173 = rsqrt_50 = None
        to_107 = hidden_states_174.to(torch.bfloat16)
        hidden_states_174 = None
        mul_163 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_
            * to_107
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_107
        ) = None
        key_states_12 = mul_163.transpose(1, 2)
        mul_163 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_170,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_170 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_38 = linear_86.view((1, 20, -1, 128))
        linear_86 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_164 = query_states_12 * cos_15
        x1_24 = query_states_12[(Ellipsis, slice(None, 64, None))]
        x2_24 = query_states_12[(Ellipsis, slice(64, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_165 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_164 + mul_165
        mul_164 = mul_165 = None
        mul_166 = key_states_12 * cos_15
        cos_15 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 64, None))]
        x2_25 = key_states_12[(Ellipsis, slice(64, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_167 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_166 + mul_167
        mul_166 = mul_167 = None
        getitem_90 = k_embed_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_12 = None
        hidden_states_175 = getitem_90.expand(1, 8, 2, 20, 128)
        getitem_90 = None
        key_12 = hidden_states_175.reshape(1, 16, 20, 128)
        hidden_states_175 = None
        getitem_91 = value_states_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_12 = None
        hidden_states_176 = getitem_91.expand(1, 8, 2, 20, 128)
        getitem_91 = None
        value_12 = hidden_states_176.reshape(1, 16, 20, 128)
        hidden_states_176 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            q_embed_12,
            key_12,
            value_12,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_12 = key_12 = value_12 = attention_mask_13 = None
        transpose_52 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_52.contiguous()
        transpose_52 = None
        reshape_38 = attn_output_49.reshape(1, 20, -1)
        attn_output_49 = None
        attn_output_50 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_177 = hidden_states_167 + attn_output_51
        hidden_states_167 = attn_output_51 = None
        hidden_states_178 = hidden_states_177.to(torch.float32)
        pow_52 = hidden_states_178.pow(2)
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_102 = variance_51 + 1e-06
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_179 = hidden_states_178 * rsqrt_51
        hidden_states_178 = rsqrt_51 = None
        to_109 = hidden_states_179.to(torch.bfloat16)
        hidden_states_179 = None
        hidden_states_180 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_109
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_109
        ) = None
        linear_88 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_88, inplace=False)
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_180,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_180 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_170 = silu_12 * linear_89
        silu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_170,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_170 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_181 = hidden_states_177 + down_proj_12
        hidden_states_177 = down_proj_12 = None
        hidden_states_182 = hidden_states_181.to(torch.float32)
        pow_53 = hidden_states_182.pow(2)
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_104 = variance_52 + 1e-06
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_104)
        add_104 = None
        hidden_states_183 = hidden_states_182 * rsqrt_52
        hidden_states_182 = rsqrt_52 = None
        to_111 = hidden_states_183.to(torch.bfloat16)
        hidden_states_183 = None
        hidden_states_184 = (
            l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
            * to_111
        )
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            to_111
        ) = None
        linear_91 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_39 = linear_91.view((1, 20, -1, 128))
        linear_91 = None
        hidden_states_185 = view_39.to(torch.float32)
        view_39 = None
        pow_54 = hidden_states_185.pow(2)
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_105 = variance_53 + 1e-06
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_105)
        add_105 = None
        hidden_states_186 = hidden_states_185 * rsqrt_53
        hidden_states_185 = rsqrt_53 = None
        to_113 = hidden_states_186.to(torch.bfloat16)
        hidden_states_186 = None
        mul_174 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_
            * to_113
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_113
        ) = None
        query_states_13 = mul_174.transpose(1, 2)
        mul_174 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_92.view((1, 20, -1, 128))
        linear_92 = None
        hidden_states_187 = view_40.to(torch.float32)
        view_40 = None
        pow_55 = hidden_states_187.pow(2)
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_106 = variance_54 + 1e-06
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_106)
        add_106 = None
        hidden_states_188 = hidden_states_187 * rsqrt_54
        hidden_states_187 = rsqrt_54 = None
        to_115 = hidden_states_188.to(torch.bfloat16)
        hidden_states_188 = None
        mul_176 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_
            * to_115
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_115
        ) = None
        key_states_13 = mul_176.transpose(1, 2)
        mul_176 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_184 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_41 = linear_93.view((1, 20, -1, 128))
        linear_93 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_177 = query_states_13 * cos_16
        x1_26 = query_states_13[(Ellipsis, slice(None, 64, None))]
        x2_26 = query_states_13[(Ellipsis, slice(64, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_178 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_177 + mul_178
        mul_177 = mul_178 = None
        mul_179 = key_states_13 * cos_16
        cos_16 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 64, None))]
        x2_27 = key_states_13[(Ellipsis, slice(64, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_180 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_179 + mul_180
        mul_179 = mul_180 = None
        getitem_97 = k_embed_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_13 = None
        hidden_states_189 = getitem_97.expand(1, 8, 2, 20, 128)
        getitem_97 = None
        key_13 = hidden_states_189.reshape(1, 16, 20, 128)
        hidden_states_189 = None
        getitem_98 = value_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_13 = None
        hidden_states_190 = getitem_98.expand(1, 8, 2, 20, 128)
        getitem_98 = None
        value_13 = hidden_states_190.reshape(1, 16, 20, 128)
        hidden_states_190 = None
        attention_mask_14 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            q_embed_13,
            key_13,
            value_13,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_13 = key_13 = value_13 = attention_mask_14 = None
        transpose_56 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_56.contiguous()
        transpose_56 = None
        reshape_41 = attn_output_53.reshape(1, 20, -1)
        attn_output_53 = None
        attn_output_54 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_191 = hidden_states_181 + attn_output_55
        hidden_states_181 = attn_output_55 = None
        hidden_states_192 = hidden_states_191.to(torch.float32)
        pow_56 = hidden_states_192.pow(2)
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_110 = variance_55 + 1e-06
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_110)
        add_110 = None
        hidden_states_193 = hidden_states_192 * rsqrt_55
        hidden_states_192 = rsqrt_55 = None
        to_117 = hidden_states_193.to(torch.bfloat16)
        hidden_states_193 = None
        hidden_states_194 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_117
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_117
        ) = None
        linear_95 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_95, inplace=False)
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_194,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_194 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_183 = silu_13 * linear_96
        silu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_183,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_183 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_195 = hidden_states_191 + down_proj_13
        hidden_states_191 = down_proj_13 = None
        hidden_states_196 = hidden_states_195.to(torch.float32)
        pow_57 = hidden_states_196.pow(2)
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_112 = variance_56 + 1e-06
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_197 = hidden_states_196 * rsqrt_56
        hidden_states_196 = rsqrt_56 = None
        to_119 = hidden_states_197.to(torch.bfloat16)
        hidden_states_197 = None
        hidden_states_198 = (
            l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
            * to_119
        )
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            to_119
        ) = None
        linear_98 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_42 = linear_98.view((1, 20, -1, 128))
        linear_98 = None
        hidden_states_199 = view_42.to(torch.float32)
        view_42 = None
        pow_58 = hidden_states_199.pow(2)
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_113 = variance_57 + 1e-06
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_113)
        add_113 = None
        hidden_states_200 = hidden_states_199 * rsqrt_57
        hidden_states_199 = rsqrt_57 = None
        to_121 = hidden_states_200.to(torch.bfloat16)
        hidden_states_200 = None
        mul_187 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_
            * to_121
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_121
        ) = None
        query_states_14 = mul_187.transpose(1, 2)
        mul_187 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_99.view((1, 20, -1, 128))
        linear_99 = None
        hidden_states_201 = view_43.to(torch.float32)
        view_43 = None
        pow_59 = hidden_states_201.pow(2)
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_114 = variance_58 + 1e-06
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_114)
        add_114 = None
        hidden_states_202 = hidden_states_201 * rsqrt_58
        hidden_states_201 = rsqrt_58 = None
        to_123 = hidden_states_202.to(torch.bfloat16)
        hidden_states_202 = None
        mul_189 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_
            * to_123
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_123
        ) = None
        key_states_14 = mul_189.transpose(1, 2)
        mul_189 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_198 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_44 = linear_100.view((1, 20, -1, 128))
        linear_100 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_190 = query_states_14 * cos_17
        x1_28 = query_states_14[(Ellipsis, slice(None, 64, None))]
        x2_28 = query_states_14[(Ellipsis, slice(64, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_191 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_190 + mul_191
        mul_190 = mul_191 = None
        mul_192 = key_states_14 * cos_17
        cos_17 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 64, None))]
        x2_29 = key_states_14[(Ellipsis, slice(64, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_193 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_192 + mul_193
        mul_192 = mul_193 = None
        getitem_104 = k_embed_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_14 = None
        hidden_states_203 = getitem_104.expand(1, 8, 2, 20, 128)
        getitem_104 = None
        key_14 = hidden_states_203.reshape(1, 16, 20, 128)
        hidden_states_203 = None
        getitem_105 = value_states_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_14 = None
        hidden_states_204 = getitem_105.expand(1, 8, 2, 20, 128)
        getitem_105 = None
        value_14 = hidden_states_204.reshape(1, 16, 20, 128)
        hidden_states_204 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            q_embed_14,
            key_14,
            value_14,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_14 = key_14 = value_14 = attention_mask_15 = None
        transpose_60 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_60.contiguous()
        transpose_60 = None
        reshape_44 = attn_output_57.reshape(1, 20, -1)
        attn_output_57 = None
        attn_output_58 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_205 = hidden_states_195 + attn_output_59
        hidden_states_195 = attn_output_59 = None
        hidden_states_206 = hidden_states_205.to(torch.float32)
        pow_60 = hidden_states_206.pow(2)
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_118 = variance_59 + 1e-06
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_207 = hidden_states_206 * rsqrt_59
        hidden_states_206 = rsqrt_59 = None
        to_125 = hidden_states_207.to(torch.bfloat16)
        hidden_states_207 = None
        hidden_states_208 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_125
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_125
        ) = None
        linear_102 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_102, inplace=False)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_208 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_196 = silu_14 * linear_103
        silu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_196,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_196 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_209 = hidden_states_205 + down_proj_14
        hidden_states_205 = down_proj_14 = None
        hidden_states_210 = hidden_states_209.to(torch.float32)
        pow_61 = hidden_states_210.pow(2)
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_120 = variance_60 + 1e-06
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_211 = hidden_states_210 * rsqrt_60
        hidden_states_210 = rsqrt_60 = None
        to_127 = hidden_states_211.to(torch.bfloat16)
        hidden_states_211 = None
        hidden_states_212 = (
            l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
            * to_127
        )
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            to_127
        ) = None
        linear_105 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_45 = linear_105.view((1, 20, -1, 128))
        linear_105 = None
        hidden_states_213 = view_45.to(torch.float32)
        view_45 = None
        pow_62 = hidden_states_213.pow(2)
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_121 = variance_61 + 1e-06
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_121)
        add_121 = None
        hidden_states_214 = hidden_states_213 * rsqrt_61
        hidden_states_213 = rsqrt_61 = None
        to_129 = hidden_states_214.to(torch.bfloat16)
        hidden_states_214 = None
        mul_200 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_
            * to_129
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_129
        ) = None
        query_states_15 = mul_200.transpose(1, 2)
        mul_200 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_106.view((1, 20, -1, 128))
        linear_106 = None
        hidden_states_215 = view_46.to(torch.float32)
        view_46 = None
        pow_63 = hidden_states_215.pow(2)
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_122 = variance_62 + 1e-06
        variance_62 = None
        rsqrt_62 = torch.rsqrt(add_122)
        add_122 = None
        hidden_states_216 = hidden_states_215 * rsqrt_62
        hidden_states_215 = rsqrt_62 = None
        to_131 = hidden_states_216.to(torch.bfloat16)
        hidden_states_216 = None
        mul_202 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_
            * to_131
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_131
        ) = None
        key_states_15 = mul_202.transpose(1, 2)
        mul_202 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_212 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_47 = linear_107.view((1, 20, -1, 128))
        linear_107 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_203 = query_states_15 * cos_18
        x1_30 = query_states_15[(Ellipsis, slice(None, 64, None))]
        x2_30 = query_states_15[(Ellipsis, slice(64, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_204 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_203 + mul_204
        mul_203 = mul_204 = None
        mul_205 = key_states_15 * cos_18
        cos_18 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 64, None))]
        x2_31 = key_states_15[(Ellipsis, slice(64, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_206 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_205 + mul_206
        mul_205 = mul_206 = None
        getitem_111 = k_embed_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_15 = None
        hidden_states_217 = getitem_111.expand(1, 8, 2, 20, 128)
        getitem_111 = None
        key_15 = hidden_states_217.reshape(1, 16, 20, 128)
        hidden_states_217 = None
        getitem_112 = value_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_15 = None
        hidden_states_218 = getitem_112.expand(1, 8, 2, 20, 128)
        getitem_112 = None
        value_15 = hidden_states_218.reshape(1, 16, 20, 128)
        hidden_states_218 = None
        attention_mask_16 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            q_embed_15,
            key_15,
            value_15,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_15 = key_15 = value_15 = attention_mask_16 = None
        transpose_64 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_64.contiguous()
        transpose_64 = None
        reshape_47 = attn_output_61.reshape(1, 20, -1)
        attn_output_61 = None
        attn_output_62 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_219 = hidden_states_209 + attn_output_63
        hidden_states_209 = attn_output_63 = None
        hidden_states_220 = hidden_states_219.to(torch.float32)
        pow_64 = hidden_states_220.pow(2)
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_126 = variance_63 + 1e-06
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_221 = hidden_states_220 * rsqrt_63
        hidden_states_220 = rsqrt_63 = None
        to_133 = hidden_states_221.to(torch.bfloat16)
        hidden_states_221 = None
        hidden_states_222 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_133
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_133
        ) = None
        linear_109 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_109, inplace=False)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_222 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_209 = silu_15 * linear_110
        silu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_209,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_209 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_223 = hidden_states_219 + down_proj_15
        hidden_states_219 = down_proj_15 = None
        hidden_states_224 = hidden_states_223.to(torch.float32)
        pow_65 = hidden_states_224.pow(2)
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_128 = variance_64 + 1e-06
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_128)
        add_128 = None
        hidden_states_225 = hidden_states_224 * rsqrt_64
        hidden_states_224 = rsqrt_64 = None
        to_135 = hidden_states_225.to(torch.bfloat16)
        hidden_states_225 = None
        hidden_states_226 = (
            l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
            * to_135
        )
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            to_135
        ) = None
        linear_112 = torch._C._nn.linear(
            hidden_states_226,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_48 = linear_112.view((1, 20, -1, 128))
        linear_112 = None
        hidden_states_227 = view_48.to(torch.float32)
        view_48 = None
        pow_66 = hidden_states_227.pow(2)
        variance_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_129 = variance_65 + 1e-06
        variance_65 = None
        rsqrt_65 = torch.rsqrt(add_129)
        add_129 = None
        hidden_states_228 = hidden_states_227 * rsqrt_65
        hidden_states_227 = rsqrt_65 = None
        to_137 = hidden_states_228.to(torch.bfloat16)
        hidden_states_228 = None
        mul_213 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_
            * to_137
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_137
        ) = None
        query_states_16 = mul_213.transpose(1, 2)
        mul_213 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_226,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_113.view((1, 20, -1, 128))
        linear_113 = None
        hidden_states_229 = view_49.to(torch.float32)
        view_49 = None
        pow_67 = hidden_states_229.pow(2)
        variance_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_130 = variance_66 + 1e-06
        variance_66 = None
        rsqrt_66 = torch.rsqrt(add_130)
        add_130 = None
        hidden_states_230 = hidden_states_229 * rsqrt_66
        hidden_states_229 = rsqrt_66 = None
        to_139 = hidden_states_230.to(torch.bfloat16)
        hidden_states_230 = None
        mul_215 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_
            * to_139
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_139
        ) = None
        key_states_16 = mul_215.transpose(1, 2)
        mul_215 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_226,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_226 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_50 = linear_114.view((1, 20, -1, 128))
        linear_114 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_216 = query_states_16 * cos_19
        x1_32 = query_states_16[(Ellipsis, slice(None, 64, None))]
        x2_32 = query_states_16[(Ellipsis, slice(64, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_217 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_216 + mul_217
        mul_216 = mul_217 = None
        mul_218 = key_states_16 * cos_19
        cos_19 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 64, None))]
        x2_33 = key_states_16[(Ellipsis, slice(64, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_219 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_218 + mul_219
        mul_218 = mul_219 = None
        getitem_118 = k_embed_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_16 = None
        hidden_states_231 = getitem_118.expand(1, 8, 2, 20, 128)
        getitem_118 = None
        key_16 = hidden_states_231.reshape(1, 16, 20, 128)
        hidden_states_231 = None
        getitem_119 = value_states_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_16 = None
        hidden_states_232 = getitem_119.expand(1, 8, 2, 20, 128)
        getitem_119 = None
        value_16 = hidden_states_232.reshape(1, 16, 20, 128)
        hidden_states_232 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            q_embed_16,
            key_16,
            value_16,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_16 = key_16 = value_16 = attention_mask_17 = None
        transpose_68 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_68.contiguous()
        transpose_68 = None
        reshape_50 = attn_output_65.reshape(1, 20, -1)
        attn_output_65 = None
        attn_output_66 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_233 = hidden_states_223 + attn_output_67
        hidden_states_223 = attn_output_67 = None
        hidden_states_234 = hidden_states_233.to(torch.float32)
        pow_68 = hidden_states_234.pow(2)
        variance_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_134 = variance_67 + 1e-06
        variance_67 = None
        rsqrt_67 = torch.rsqrt(add_134)
        add_134 = None
        hidden_states_235 = hidden_states_234 * rsqrt_67
        hidden_states_234 = rsqrt_67 = None
        to_141 = hidden_states_235.to(torch.bfloat16)
        hidden_states_235 = None
        hidden_states_236 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_141
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_141
        ) = None
        linear_116 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_116, inplace=False)
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_236,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_236 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_222 = silu_16 * linear_117
        silu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_222,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_222 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_237 = hidden_states_233 + down_proj_16
        hidden_states_233 = down_proj_16 = None
        hidden_states_238 = hidden_states_237.to(torch.float32)
        pow_69 = hidden_states_238.pow(2)
        variance_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_136 = variance_68 + 1e-06
        variance_68 = None
        rsqrt_68 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_239 = hidden_states_238 * rsqrt_68
        hidden_states_238 = rsqrt_68 = None
        to_143 = hidden_states_239.to(torch.bfloat16)
        hidden_states_239 = None
        hidden_states_240 = (
            l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
            * to_143
        )
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            to_143
        ) = None
        linear_119 = torch._C._nn.linear(
            hidden_states_240,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_51 = linear_119.view((1, 20, -1, 128))
        linear_119 = None
        hidden_states_241 = view_51.to(torch.float32)
        view_51 = None
        pow_70 = hidden_states_241.pow(2)
        variance_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_137 = variance_69 + 1e-06
        variance_69 = None
        rsqrt_69 = torch.rsqrt(add_137)
        add_137 = None
        hidden_states_242 = hidden_states_241 * rsqrt_69
        hidden_states_241 = rsqrt_69 = None
        to_145 = hidden_states_242.to(torch.bfloat16)
        hidden_states_242 = None
        mul_226 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_
            * to_145
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_145
        ) = None
        query_states_17 = mul_226.transpose(1, 2)
        mul_226 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_240,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_120.view((1, 20, -1, 128))
        linear_120 = None
        hidden_states_243 = view_52.to(torch.float32)
        view_52 = None
        pow_71 = hidden_states_243.pow(2)
        variance_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_138 = variance_70 + 1e-06
        variance_70 = None
        rsqrt_70 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_244 = hidden_states_243 * rsqrt_70
        hidden_states_243 = rsqrt_70 = None
        to_147 = hidden_states_244.to(torch.bfloat16)
        hidden_states_244 = None
        mul_228 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_
            * to_147
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_147
        ) = None
        key_states_17 = mul_228.transpose(1, 2)
        mul_228 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_240,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_240 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_53 = linear_121.view((1, 20, -1, 128))
        linear_121 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_229 = query_states_17 * cos_20
        x1_34 = query_states_17[(Ellipsis, slice(None, 64, None))]
        x2_34 = query_states_17[(Ellipsis, slice(64, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_230 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_229 + mul_230
        mul_229 = mul_230 = None
        mul_231 = key_states_17 * cos_20
        cos_20 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 64, None))]
        x2_35 = key_states_17[(Ellipsis, slice(64, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_232 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_231 + mul_232
        mul_231 = mul_232 = None
        getitem_125 = k_embed_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_17 = None
        hidden_states_245 = getitem_125.expand(1, 8, 2, 20, 128)
        getitem_125 = None
        key_17 = hidden_states_245.reshape(1, 16, 20, 128)
        hidden_states_245 = None
        getitem_126 = value_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_17 = None
        hidden_states_246 = getitem_126.expand(1, 8, 2, 20, 128)
        getitem_126 = None
        value_17 = hidden_states_246.reshape(1, 16, 20, 128)
        hidden_states_246 = None
        attention_mask_18 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            q_embed_17,
            key_17,
            value_17,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_17 = key_17 = value_17 = attention_mask_18 = None
        transpose_72 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_72.contiguous()
        transpose_72 = None
        reshape_53 = attn_output_69.reshape(1, 20, -1)
        attn_output_69 = None
        attn_output_70 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_247 = hidden_states_237 + attn_output_71
        hidden_states_237 = attn_output_71 = None
        hidden_states_248 = hidden_states_247.to(torch.float32)
        pow_72 = hidden_states_248.pow(2)
        variance_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_142 = variance_71 + 1e-06
        variance_71 = None
        rsqrt_71 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_249 = hidden_states_248 * rsqrt_71
        hidden_states_248 = rsqrt_71 = None
        to_149 = hidden_states_249.to(torch.bfloat16)
        hidden_states_249 = None
        hidden_states_250 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_149
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_149
        ) = None
        linear_123 = torch._C._nn.linear(
            hidden_states_250,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_123, inplace=False)
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_250,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_250 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_235 = silu_17 * linear_124
        silu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_235,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_235 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_251 = hidden_states_247 + down_proj_17
        hidden_states_247 = down_proj_17 = None
        hidden_states_252 = hidden_states_251.to(torch.float32)
        pow_73 = hidden_states_252.pow(2)
        variance_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_144 = variance_72 + 1e-06
        variance_72 = None
        rsqrt_72 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_253 = hidden_states_252 * rsqrt_72
        hidden_states_252 = rsqrt_72 = None
        to_151 = hidden_states_253.to(torch.bfloat16)
        hidden_states_253 = None
        hidden_states_254 = (
            l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
            * to_151
        )
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            to_151
        ) = None
        linear_126 = torch._C._nn.linear(
            hidden_states_254,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_54 = linear_126.view((1, 20, -1, 128))
        linear_126 = None
        hidden_states_255 = view_54.to(torch.float32)
        view_54 = None
        pow_74 = hidden_states_255.pow(2)
        variance_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_145 = variance_73 + 1e-06
        variance_73 = None
        rsqrt_73 = torch.rsqrt(add_145)
        add_145 = None
        hidden_states_256 = hidden_states_255 * rsqrt_73
        hidden_states_255 = rsqrt_73 = None
        to_153 = hidden_states_256.to(torch.bfloat16)
        hidden_states_256 = None
        mul_239 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_
            * to_153
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_153
        ) = None
        query_states_18 = mul_239.transpose(1, 2)
        mul_239 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_254,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_55 = linear_127.view((1, 20, -1, 128))
        linear_127 = None
        hidden_states_257 = view_55.to(torch.float32)
        view_55 = None
        pow_75 = hidden_states_257.pow(2)
        variance_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_146 = variance_74 + 1e-06
        variance_74 = None
        rsqrt_74 = torch.rsqrt(add_146)
        add_146 = None
        hidden_states_258 = hidden_states_257 * rsqrt_74
        hidden_states_257 = rsqrt_74 = None
        to_155 = hidden_states_258.to(torch.bfloat16)
        hidden_states_258 = None
        mul_241 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_
            * to_155
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_155
        ) = None
        key_states_18 = mul_241.transpose(1, 2)
        mul_241 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_254,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_254 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_56 = linear_128.view((1, 20, -1, 128))
        linear_128 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_242 = query_states_18 * cos_21
        x1_36 = query_states_18[(Ellipsis, slice(None, 64, None))]
        x2_36 = query_states_18[(Ellipsis, slice(64, None, None))]
        query_states_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_37 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_243 = cat_37 * sin_21
        cat_37 = None
        q_embed_18 = mul_242 + mul_243
        mul_242 = mul_243 = None
        mul_244 = key_states_18 * cos_21
        cos_21 = None
        x1_37 = key_states_18[(Ellipsis, slice(None, 64, None))]
        x2_37 = key_states_18[(Ellipsis, slice(64, None, None))]
        key_states_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_38 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_245 = cat_38 * sin_21
        cat_38 = sin_21 = None
        k_embed_18 = mul_244 + mul_245
        mul_244 = mul_245 = None
        getitem_132 = k_embed_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_18 = None
        hidden_states_259 = getitem_132.expand(1, 8, 2, 20, 128)
        getitem_132 = None
        key_18 = hidden_states_259.reshape(1, 16, 20, 128)
        hidden_states_259 = None
        getitem_133 = value_states_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_18 = None
        hidden_states_260 = getitem_133.expand(1, 8, 2, 20, 128)
        getitem_133 = None
        value_18 = hidden_states_260.reshape(1, 16, 20, 128)
        hidden_states_260 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            q_embed_18,
            key_18,
            value_18,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_18 = key_18 = value_18 = attention_mask_19 = None
        transpose_76 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_76.contiguous()
        transpose_76 = None
        reshape_56 = attn_output_73.reshape(1, 20, -1)
        attn_output_73 = None
        attn_output_74 = reshape_56.contiguous()
        reshape_56 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_261 = hidden_states_251 + attn_output_75
        hidden_states_251 = attn_output_75 = None
        hidden_states_262 = hidden_states_261.to(torch.float32)
        pow_76 = hidden_states_262.pow(2)
        variance_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_150 = variance_75 + 1e-06
        variance_75 = None
        rsqrt_75 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_263 = hidden_states_262 * rsqrt_75
        hidden_states_262 = rsqrt_75 = None
        to_157 = hidden_states_263.to(torch.bfloat16)
        hidden_states_263 = None
        hidden_states_264 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
            * to_157
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            to_157
        ) = None
        linear_130 = torch._C._nn.linear(
            hidden_states_264,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_130, inplace=False)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_264,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_264 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_248 = silu_18 * linear_131
        silu_18 = linear_131 = None
        down_proj_18 = torch._C._nn.linear(
            mul_248,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_248 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_265 = hidden_states_261 + down_proj_18
        hidden_states_261 = down_proj_18 = None
        hidden_states_266 = hidden_states_265.to(torch.float32)
        pow_77 = hidden_states_266.pow(2)
        variance_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_152 = variance_76 + 1e-06
        variance_76 = None
        rsqrt_76 = torch.rsqrt(add_152)
        add_152 = None
        hidden_states_267 = hidden_states_266 * rsqrt_76
        hidden_states_266 = rsqrt_76 = None
        to_159 = hidden_states_267.to(torch.bfloat16)
        hidden_states_267 = None
        hidden_states_268 = (
            l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
            * to_159
        )
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            to_159
        ) = None
        linear_133 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_57 = linear_133.view((1, 20, -1, 128))
        linear_133 = None
        hidden_states_269 = view_57.to(torch.float32)
        view_57 = None
        pow_78 = hidden_states_269.pow(2)
        variance_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_153 = variance_77 + 1e-06
        variance_77 = None
        rsqrt_77 = torch.rsqrt(add_153)
        add_153 = None
        hidden_states_270 = hidden_states_269 * rsqrt_77
        hidden_states_269 = rsqrt_77 = None
        to_161 = hidden_states_270.to(torch.bfloat16)
        hidden_states_270 = None
        mul_252 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_
            * to_161
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_161
        ) = None
        query_states_19 = mul_252.transpose(1, 2)
        mul_252 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_58 = linear_134.view((1, 20, -1, 128))
        linear_134 = None
        hidden_states_271 = view_58.to(torch.float32)
        view_58 = None
        pow_79 = hidden_states_271.pow(2)
        variance_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        add_154 = variance_78 + 1e-06
        variance_78 = None
        rsqrt_78 = torch.rsqrt(add_154)
        add_154 = None
        hidden_states_272 = hidden_states_271 * rsqrt_78
        hidden_states_271 = rsqrt_78 = None
        to_163 = hidden_states_272.to(torch.bfloat16)
        hidden_states_272 = None
        mul_254 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_
            * to_163
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_163
        ) = None
        key_states_19 = mul_254.transpose(1, 2)
        mul_254 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_268 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_59 = linear_135.view((1, 20, -1, 128))
        linear_135 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_255 = query_states_19 * cos_22
        x1_38 = query_states_19[(Ellipsis, slice(None, 64, None))]
        x2_38 = query_states_19[(Ellipsis, slice(64, None, None))]
        query_states_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_39 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_256 = cat_39 * sin_22
        cat_39 = None
        q_embed_19 = mul_255 + mul_256
        mul_255 = mul_256 = None
        mul_257 = key_states_19 * cos_22
        cos_22 = None
        x1_39 = key_states_19[(Ellipsis, slice(None, 64, None))]
        x2_39 = key_states_19[(Ellipsis, slice(64, None, None))]
        key_states_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_40 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_258 = cat_40 * sin_22
        cat_40 = sin_22 = None
        k_embed_19 = mul_257 + mul_258
        mul_257 = mul_258 = None
        getitem_139 = k_embed_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_19 = None
        hidden_states_273 = getitem_139.expand(1, 8, 2, 20, 128)
        getitem_139 = None
        key_19 = hidden_states_273.reshape(1, 16, 20, 128)
        hidden_states_273 = None
        getitem_140 = value_states_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_19 = None
        hidden_states_274 = getitem_140.expand(1, 8, 2, 20, 128)
        getitem_140 = None
        value_19 = hidden_states_274.reshape(1, 16, 20, 128)
        hidden_states_274 = None
        attention_mask_20 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            q_embed_19,
            key_19,
            value_19,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_19 = key_19 = value_19 = attention_mask_20 = None
        transpose_80 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_80.contiguous()
        transpose_80 = None
        reshape_59 = attn_output_77.reshape(1, 20, -1)
        attn_output_77 = None
        attn_output_78 = reshape_59.contiguous()
        reshape_59 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_275 = hidden_states_265 + attn_output_79
        hidden_states_265 = attn_output_79 = None
        hidden_states_276 = hidden_states_275.to(torch.float32)
        pow_80 = hidden_states_276.pow(2)
        variance_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_158 = variance_79 + 1e-06
        variance_79 = None
        rsqrt_79 = torch.rsqrt(add_158)
        add_158 = None
        hidden_states_277 = hidden_states_276 * rsqrt_79
        hidden_states_276 = rsqrt_79 = None
        to_165 = hidden_states_277.to(torch.bfloat16)
        hidden_states_277 = None
        hidden_states_278 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
            * to_165
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            to_165
        ) = None
        linear_137 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_137, inplace=False)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_278 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_261 = silu_19 * linear_138
        silu_19 = linear_138 = None
        down_proj_19 = torch._C._nn.linear(
            mul_261,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_261 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_279 = hidden_states_275 + down_proj_19
        hidden_states_275 = down_proj_19 = None
        hidden_states_280 = hidden_states_279.to(torch.float32)
        pow_81 = hidden_states_280.pow(2)
        variance_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_160 = variance_80 + 1e-06
        variance_80 = None
        rsqrt_80 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_281 = hidden_states_280 * rsqrt_80
        hidden_states_280 = rsqrt_80 = None
        to_167 = hidden_states_281.to(torch.bfloat16)
        hidden_states_281 = None
        hidden_states_282 = (
            l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
            * to_167
        )
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            to_167
        ) = None
        linear_140 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_60 = linear_140.view((1, 20, -1, 128))
        linear_140 = None
        hidden_states_283 = view_60.to(torch.float32)
        view_60 = None
        pow_82 = hidden_states_283.pow(2)
        variance_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_161 = variance_81 + 1e-06
        variance_81 = None
        rsqrt_81 = torch.rsqrt(add_161)
        add_161 = None
        hidden_states_284 = hidden_states_283 * rsqrt_81
        hidden_states_283 = rsqrt_81 = None
        to_169 = hidden_states_284.to(torch.bfloat16)
        hidden_states_284 = None
        mul_265 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_
            * to_169
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_169
        ) = None
        query_states_20 = mul_265.transpose(1, 2)
        mul_265 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_61 = linear_141.view((1, 20, -1, 128))
        linear_141 = None
        hidden_states_285 = view_61.to(torch.float32)
        view_61 = None
        pow_83 = hidden_states_285.pow(2)
        variance_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        add_162 = variance_82 + 1e-06
        variance_82 = None
        rsqrt_82 = torch.rsqrt(add_162)
        add_162 = None
        hidden_states_286 = hidden_states_285 * rsqrt_82
        hidden_states_285 = rsqrt_82 = None
        to_171 = hidden_states_286.to(torch.bfloat16)
        hidden_states_286 = None
        mul_267 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_
            * to_171
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_171
        ) = None
        key_states_20 = mul_267.transpose(1, 2)
        mul_267 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_282 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_62 = linear_142.view((1, 20, -1, 128))
        linear_142 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_268 = query_states_20 * cos_23
        x1_40 = query_states_20[(Ellipsis, slice(None, 64, None))]
        x2_40 = query_states_20[(Ellipsis, slice(64, None, None))]
        query_states_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_41 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_269 = cat_41 * sin_23
        cat_41 = None
        q_embed_20 = mul_268 + mul_269
        mul_268 = mul_269 = None
        mul_270 = key_states_20 * cos_23
        cos_23 = None
        x1_41 = key_states_20[(Ellipsis, slice(None, 64, None))]
        x2_41 = key_states_20[(Ellipsis, slice(64, None, None))]
        key_states_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_42 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_271 = cat_42 * sin_23
        cat_42 = sin_23 = None
        k_embed_20 = mul_270 + mul_271
        mul_270 = mul_271 = None
        getitem_146 = k_embed_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_20 = None
        hidden_states_287 = getitem_146.expand(1, 8, 2, 20, 128)
        getitem_146 = None
        key_20 = hidden_states_287.reshape(1, 16, 20, 128)
        hidden_states_287 = None
        getitem_147 = value_states_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_20 = None
        hidden_states_288 = getitem_147.expand(1, 8, 2, 20, 128)
        getitem_147 = None
        value_20 = hidden_states_288.reshape(1, 16, 20, 128)
        hidden_states_288 = None
        attention_mask_21 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            q_embed_20,
            key_20,
            value_20,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_20 = key_20 = value_20 = attention_mask_21 = None
        transpose_84 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_84.contiguous()
        transpose_84 = None
        reshape_62 = attn_output_81.reshape(1, 20, -1)
        attn_output_81 = None
        attn_output_82 = reshape_62.contiguous()
        reshape_62 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_289 = hidden_states_279 + attn_output_83
        hidden_states_279 = attn_output_83 = None
        hidden_states_290 = hidden_states_289.to(torch.float32)
        pow_84 = hidden_states_290.pow(2)
        variance_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_166 = variance_83 + 1e-06
        variance_83 = None
        rsqrt_83 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_291 = hidden_states_290 * rsqrt_83
        hidden_states_290 = rsqrt_83 = None
        to_173 = hidden_states_291.to(torch.bfloat16)
        hidden_states_291 = None
        hidden_states_292 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
            * to_173
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            to_173
        ) = None
        linear_144 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_144, inplace=False)
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_292 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_274 = silu_20 * linear_145
        silu_20 = linear_145 = None
        down_proj_20 = torch._C._nn.linear(
            mul_274,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_274 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_293 = hidden_states_289 + down_proj_20
        hidden_states_289 = down_proj_20 = None
        hidden_states_294 = hidden_states_293.to(torch.float32)
        pow_85 = hidden_states_294.pow(2)
        variance_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_168 = variance_84 + 1e-06
        variance_84 = None
        rsqrt_84 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_295 = hidden_states_294 * rsqrt_84
        hidden_states_294 = rsqrt_84 = None
        to_175 = hidden_states_295.to(torch.bfloat16)
        hidden_states_295 = None
        hidden_states_296 = (
            l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
            * to_175
        )
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            to_175
        ) = None
        linear_147 = torch._C._nn.linear(
            hidden_states_296,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_63 = linear_147.view((1, 20, -1, 128))
        linear_147 = None
        hidden_states_297 = view_63.to(torch.float32)
        view_63 = None
        pow_86 = hidden_states_297.pow(2)
        variance_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_169 = variance_85 + 1e-06
        variance_85 = None
        rsqrt_85 = torch.rsqrt(add_169)
        add_169 = None
        hidden_states_298 = hidden_states_297 * rsqrt_85
        hidden_states_297 = rsqrt_85 = None
        to_177 = hidden_states_298.to(torch.bfloat16)
        hidden_states_298 = None
        mul_278 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_
            * to_177
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_177
        ) = None
        query_states_21 = mul_278.transpose(1, 2)
        mul_278 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_296,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_64 = linear_148.view((1, 20, -1, 128))
        linear_148 = None
        hidden_states_299 = view_64.to(torch.float32)
        view_64 = None
        pow_87 = hidden_states_299.pow(2)
        variance_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        add_170 = variance_86 + 1e-06
        variance_86 = None
        rsqrt_86 = torch.rsqrt(add_170)
        add_170 = None
        hidden_states_300 = hidden_states_299 * rsqrt_86
        hidden_states_299 = rsqrt_86 = None
        to_179 = hidden_states_300.to(torch.bfloat16)
        hidden_states_300 = None
        mul_280 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_
            * to_179
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_179
        ) = None
        key_states_21 = mul_280.transpose(1, 2)
        mul_280 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_296,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_296 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_65 = linear_149.view((1, 20, -1, 128))
        linear_149 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_281 = query_states_21 * cos_24
        x1_42 = query_states_21[(Ellipsis, slice(None, 64, None))]
        x2_42 = query_states_21[(Ellipsis, slice(64, None, None))]
        query_states_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_43 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_282 = cat_43 * sin_24
        cat_43 = None
        q_embed_21 = mul_281 + mul_282
        mul_281 = mul_282 = None
        mul_283 = key_states_21 * cos_24
        cos_24 = None
        x1_43 = key_states_21[(Ellipsis, slice(None, 64, None))]
        x2_43 = key_states_21[(Ellipsis, slice(64, None, None))]
        key_states_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_44 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_284 = cat_44 * sin_24
        cat_44 = sin_24 = None
        k_embed_21 = mul_283 + mul_284
        mul_283 = mul_284 = None
        getitem_153 = k_embed_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_21 = None
        hidden_states_301 = getitem_153.expand(1, 8, 2, 20, 128)
        getitem_153 = None
        key_21 = hidden_states_301.reshape(1, 16, 20, 128)
        hidden_states_301 = None
        getitem_154 = value_states_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_21 = None
        hidden_states_302 = getitem_154.expand(1, 8, 2, 20, 128)
        getitem_154 = None
        value_21 = hidden_states_302.reshape(1, 16, 20, 128)
        hidden_states_302 = None
        attention_mask_22 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            q_embed_21,
            key_21,
            value_21,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_21 = key_21 = value_21 = attention_mask_22 = None
        transpose_88 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_88.contiguous()
        transpose_88 = None
        reshape_65 = attn_output_85.reshape(1, 20, -1)
        attn_output_85 = None
        attn_output_86 = reshape_65.contiguous()
        reshape_65 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_303 = hidden_states_293 + attn_output_87
        hidden_states_293 = attn_output_87 = None
        hidden_states_304 = hidden_states_303.to(torch.float32)
        pow_88 = hidden_states_304.pow(2)
        variance_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_174 = variance_87 + 1e-06
        variance_87 = None
        rsqrt_87 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_305 = hidden_states_304 * rsqrt_87
        hidden_states_304 = rsqrt_87 = None
        to_181 = hidden_states_305.to(torch.bfloat16)
        hidden_states_305 = None
        hidden_states_306 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
            * to_181
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            to_181
        ) = None
        linear_151 = torch._C._nn.linear(
            hidden_states_306,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_151, inplace=False)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_306,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_306 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_287 = silu_21 * linear_152
        silu_21 = linear_152 = None
        down_proj_21 = torch._C._nn.linear(
            mul_287,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_287 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_307 = hidden_states_303 + down_proj_21
        hidden_states_303 = down_proj_21 = None
        hidden_states_308 = hidden_states_307.to(torch.float32)
        pow_89 = hidden_states_308.pow(2)
        variance_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_176 = variance_88 + 1e-06
        variance_88 = None
        rsqrt_88 = torch.rsqrt(add_176)
        add_176 = None
        hidden_states_309 = hidden_states_308 * rsqrt_88
        hidden_states_308 = rsqrt_88 = None
        to_183 = hidden_states_309.to(torch.bfloat16)
        hidden_states_309 = None
        hidden_states_310 = (
            l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
            * to_183
        )
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            to_183
        ) = None
        linear_154 = torch._C._nn.linear(
            hidden_states_310,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_66 = linear_154.view((1, 20, -1, 128))
        linear_154 = None
        hidden_states_311 = view_66.to(torch.float32)
        view_66 = None
        pow_90 = hidden_states_311.pow(2)
        variance_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_177 = variance_89 + 1e-06
        variance_89 = None
        rsqrt_89 = torch.rsqrt(add_177)
        add_177 = None
        hidden_states_312 = hidden_states_311 * rsqrt_89
        hidden_states_311 = rsqrt_89 = None
        to_185 = hidden_states_312.to(torch.bfloat16)
        hidden_states_312 = None
        mul_291 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_
            * to_185
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_185
        ) = None
        query_states_22 = mul_291.transpose(1, 2)
        mul_291 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_310,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_67 = linear_155.view((1, 20, -1, 128))
        linear_155 = None
        hidden_states_313 = view_67.to(torch.float32)
        view_67 = None
        pow_91 = hidden_states_313.pow(2)
        variance_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        add_178 = variance_90 + 1e-06
        variance_90 = None
        rsqrt_90 = torch.rsqrt(add_178)
        add_178 = None
        hidden_states_314 = hidden_states_313 * rsqrt_90
        hidden_states_313 = rsqrt_90 = None
        to_187 = hidden_states_314.to(torch.bfloat16)
        hidden_states_314 = None
        mul_293 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_
            * to_187
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_187
        ) = None
        key_states_22 = mul_293.transpose(1, 2)
        mul_293 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_310,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_310 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_68 = linear_156.view((1, 20, -1, 128))
        linear_156 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_294 = query_states_22 * cos_25
        x1_44 = query_states_22[(Ellipsis, slice(None, 64, None))]
        x2_44 = query_states_22[(Ellipsis, slice(64, None, None))]
        query_states_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_45 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_295 = cat_45 * sin_25
        cat_45 = None
        q_embed_22 = mul_294 + mul_295
        mul_294 = mul_295 = None
        mul_296 = key_states_22 * cos_25
        cos_25 = None
        x1_45 = key_states_22[(Ellipsis, slice(None, 64, None))]
        x2_45 = key_states_22[(Ellipsis, slice(64, None, None))]
        key_states_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_46 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_297 = cat_46 * sin_25
        cat_46 = sin_25 = None
        k_embed_22 = mul_296 + mul_297
        mul_296 = mul_297 = None
        getitem_160 = k_embed_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_22 = None
        hidden_states_315 = getitem_160.expand(1, 8, 2, 20, 128)
        getitem_160 = None
        key_22 = hidden_states_315.reshape(1, 16, 20, 128)
        hidden_states_315 = None
        getitem_161 = value_states_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_22 = None
        hidden_states_316 = getitem_161.expand(1, 8, 2, 20, 128)
        getitem_161 = None
        value_22 = hidden_states_316.reshape(1, 16, 20, 128)
        hidden_states_316 = None
        attention_mask_23 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            q_embed_22,
            key_22,
            value_22,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_22 = key_22 = value_22 = attention_mask_23 = None
        transpose_92 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_92.contiguous()
        transpose_92 = None
        reshape_68 = attn_output_89.reshape(1, 20, -1)
        attn_output_89 = None
        attn_output_90 = reshape_68.contiguous()
        reshape_68 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_317 = hidden_states_307 + attn_output_91
        hidden_states_307 = attn_output_91 = None
        hidden_states_318 = hidden_states_317.to(torch.float32)
        pow_92 = hidden_states_318.pow(2)
        variance_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_182 = variance_91 + 1e-06
        variance_91 = None
        rsqrt_91 = torch.rsqrt(add_182)
        add_182 = None
        hidden_states_319 = hidden_states_318 * rsqrt_91
        hidden_states_318 = rsqrt_91 = None
        to_189 = hidden_states_319.to(torch.bfloat16)
        hidden_states_319 = None
        hidden_states_320 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
            * to_189
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            to_189
        ) = None
        linear_158 = torch._C._nn.linear(
            hidden_states_320,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_158, inplace=False)
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_320,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_320 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_300 = silu_22 * linear_159
        silu_22 = linear_159 = None
        down_proj_22 = torch._C._nn.linear(
            mul_300,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_300 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_321 = hidden_states_317 + down_proj_22
        hidden_states_317 = down_proj_22 = None
        hidden_states_322 = hidden_states_321.to(torch.float32)
        pow_93 = hidden_states_322.pow(2)
        variance_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_184 = variance_92 + 1e-06
        variance_92 = None
        rsqrt_92 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_323 = hidden_states_322 * rsqrt_92
        hidden_states_322 = rsqrt_92 = None
        to_191 = hidden_states_323.to(torch.bfloat16)
        hidden_states_323 = None
        hidden_states_324 = (
            l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
            * to_191
        )
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            to_191
        ) = None
        linear_161 = torch._C._nn.linear(
            hidden_states_324,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_69 = linear_161.view((1, 20, -1, 128))
        linear_161 = None
        hidden_states_325 = view_69.to(torch.float32)
        view_69 = None
        pow_94 = hidden_states_325.pow(2)
        variance_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_185 = variance_93 + 1e-06
        variance_93 = None
        rsqrt_93 = torch.rsqrt(add_185)
        add_185 = None
        hidden_states_326 = hidden_states_325 * rsqrt_93
        hidden_states_325 = rsqrt_93 = None
        to_193 = hidden_states_326.to(torch.bfloat16)
        hidden_states_326 = None
        mul_304 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_
            * to_193
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_193
        ) = None
        query_states_23 = mul_304.transpose(1, 2)
        mul_304 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_324,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_70 = linear_162.view((1, 20, -1, 128))
        linear_162 = None
        hidden_states_327 = view_70.to(torch.float32)
        view_70 = None
        pow_95 = hidden_states_327.pow(2)
        variance_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        add_186 = variance_94 + 1e-06
        variance_94 = None
        rsqrt_94 = torch.rsqrt(add_186)
        add_186 = None
        hidden_states_328 = hidden_states_327 * rsqrt_94
        hidden_states_327 = rsqrt_94 = None
        to_195 = hidden_states_328.to(torch.bfloat16)
        hidden_states_328 = None
        mul_306 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_
            * to_195
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_195
        ) = None
        key_states_23 = mul_306.transpose(1, 2)
        mul_306 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_324,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_324 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_71 = linear_163.view((1, 20, -1, 128))
        linear_163 = None
        value_states_23 = view_71.transpose(1, 2)
        view_71 = None
        cos_26 = cos_2.unsqueeze(1)
        sin_26 = sin_2.unsqueeze(1)
        mul_307 = query_states_23 * cos_26
        x1_46 = query_states_23[(Ellipsis, slice(None, 64, None))]
        x2_46 = query_states_23[(Ellipsis, slice(64, None, None))]
        query_states_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_47 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_308 = cat_47 * sin_26
        cat_47 = None
        q_embed_23 = mul_307 + mul_308
        mul_307 = mul_308 = None
        mul_309 = key_states_23 * cos_26
        cos_26 = None
        x1_47 = key_states_23[(Ellipsis, slice(None, 64, None))]
        x2_47 = key_states_23[(Ellipsis, slice(64, None, None))]
        key_states_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_48 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_310 = cat_48 * sin_26
        cat_48 = sin_26 = None
        k_embed_23 = mul_309 + mul_310
        mul_309 = mul_310 = None
        getitem_167 = k_embed_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_23 = None
        hidden_states_329 = getitem_167.expand(1, 8, 2, 20, 128)
        getitem_167 = None
        key_23 = hidden_states_329.reshape(1, 16, 20, 128)
        hidden_states_329 = None
        getitem_168 = value_states_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_23 = None
        hidden_states_330 = getitem_168.expand(1, 8, 2, 20, 128)
        getitem_168 = None
        value_23 = hidden_states_330.reshape(1, 16, 20, 128)
        hidden_states_330 = None
        attention_mask_24 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            q_embed_23,
            key_23,
            value_23,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_23 = key_23 = value_23 = attention_mask_24 = None
        transpose_96 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_96.contiguous()
        transpose_96 = None
        reshape_71 = attn_output_93.reshape(1, 20, -1)
        attn_output_93 = None
        attn_output_94 = reshape_71.contiguous()
        reshape_71 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_331 = hidden_states_321 + attn_output_95
        hidden_states_321 = attn_output_95 = None
        hidden_states_332 = hidden_states_331.to(torch.float32)
        pow_96 = hidden_states_332.pow(2)
        variance_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_190 = variance_95 + 1e-06
        variance_95 = None
        rsqrt_95 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_333 = hidden_states_332 * rsqrt_95
        hidden_states_332 = rsqrt_95 = None
        to_197 = hidden_states_333.to(torch.bfloat16)
        hidden_states_333 = None
        hidden_states_334 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
            * to_197
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            to_197
        ) = None
        linear_165 = torch._C._nn.linear(
            hidden_states_334,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_165, inplace=False)
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_334,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_334 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_313 = silu_23 * linear_166
        silu_23 = linear_166 = None
        down_proj_23 = torch._C._nn.linear(
            mul_313,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_313 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_335 = hidden_states_331 + down_proj_23
        hidden_states_331 = down_proj_23 = None
        hidden_states_336 = hidden_states_335.to(torch.float32)
        pow_97 = hidden_states_336.pow(2)
        variance_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_192 = variance_96 + 1e-06
        variance_96 = None
        rsqrt_96 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_337 = hidden_states_336 * rsqrt_96
        hidden_states_336 = rsqrt_96 = None
        to_199 = hidden_states_337.to(torch.bfloat16)
        hidden_states_337 = None
        hidden_states_338 = (
            l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
            * to_199
        )
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            to_199
        ) = None
        linear_168 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_72 = linear_168.view((1, 20, -1, 128))
        linear_168 = None
        hidden_states_339 = view_72.to(torch.float32)
        view_72 = None
        pow_98 = hidden_states_339.pow(2)
        variance_97 = pow_98.mean(-1, keepdim=True)
        pow_98 = None
        add_193 = variance_97 + 1e-06
        variance_97 = None
        rsqrt_97 = torch.rsqrt(add_193)
        add_193 = None
        hidden_states_340 = hidden_states_339 * rsqrt_97
        hidden_states_339 = rsqrt_97 = None
        to_201 = hidden_states_340.to(torch.bfloat16)
        hidden_states_340 = None
        mul_317 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_
            * to_201
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_201
        ) = None
        query_states_24 = mul_317.transpose(1, 2)
        mul_317 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_73 = linear_169.view((1, 20, -1, 128))
        linear_169 = None
        hidden_states_341 = view_73.to(torch.float32)
        view_73 = None
        pow_99 = hidden_states_341.pow(2)
        variance_98 = pow_99.mean(-1, keepdim=True)
        pow_99 = None
        add_194 = variance_98 + 1e-06
        variance_98 = None
        rsqrt_98 = torch.rsqrt(add_194)
        add_194 = None
        hidden_states_342 = hidden_states_341 * rsqrt_98
        hidden_states_341 = rsqrt_98 = None
        to_203 = hidden_states_342.to(torch.bfloat16)
        hidden_states_342 = None
        mul_319 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_
            * to_203
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_203
        ) = None
        key_states_24 = mul_319.transpose(1, 2)
        mul_319 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_338 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_74 = linear_170.view((1, 20, -1, 128))
        linear_170 = None
        value_states_24 = view_74.transpose(1, 2)
        view_74 = None
        cos_27 = cos_2.unsqueeze(1)
        sin_27 = sin_2.unsqueeze(1)
        mul_320 = query_states_24 * cos_27
        x1_48 = query_states_24[(Ellipsis, slice(None, 64, None))]
        x2_48 = query_states_24[(Ellipsis, slice(64, None, None))]
        query_states_24 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_49 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_321 = cat_49 * sin_27
        cat_49 = None
        q_embed_24 = mul_320 + mul_321
        mul_320 = mul_321 = None
        mul_322 = key_states_24 * cos_27
        cos_27 = None
        x1_49 = key_states_24[(Ellipsis, slice(None, 64, None))]
        x2_49 = key_states_24[(Ellipsis, slice(64, None, None))]
        key_states_24 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_50 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_323 = cat_50 * sin_27
        cat_50 = sin_27 = None
        k_embed_24 = mul_322 + mul_323
        mul_322 = mul_323 = None
        getitem_174 = k_embed_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_24 = None
        hidden_states_343 = getitem_174.expand(1, 8, 2, 20, 128)
        getitem_174 = None
        key_24 = hidden_states_343.reshape(1, 16, 20, 128)
        hidden_states_343 = None
        getitem_175 = value_states_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_24 = None
        hidden_states_344 = getitem_175.expand(1, 8, 2, 20, 128)
        getitem_175 = None
        value_24 = hidden_states_344.reshape(1, 16, 20, 128)
        hidden_states_344 = None
        attention_mask_25 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            q_embed_24,
            key_24,
            value_24,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_24 = key_24 = value_24 = attention_mask_25 = None
        transpose_100 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_100.contiguous()
        transpose_100 = None
        reshape_74 = attn_output_97.reshape(1, 20, -1)
        attn_output_97 = None
        attn_output_98 = reshape_74.contiguous()
        reshape_74 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_345 = hidden_states_335 + attn_output_99
        hidden_states_335 = attn_output_99 = None
        hidden_states_346 = hidden_states_345.to(torch.float32)
        pow_100 = hidden_states_346.pow(2)
        variance_99 = pow_100.mean(-1, keepdim=True)
        pow_100 = None
        add_198 = variance_99 + 1e-06
        variance_99 = None
        rsqrt_99 = torch.rsqrt(add_198)
        add_198 = None
        hidden_states_347 = hidden_states_346 * rsqrt_99
        hidden_states_346 = rsqrt_99 = None
        to_205 = hidden_states_347.to(torch.bfloat16)
        hidden_states_347 = None
        hidden_states_348 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
            * to_205
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            to_205
        ) = None
        linear_172 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_172, inplace=False)
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_348 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_326 = silu_24 * linear_173
        silu_24 = linear_173 = None
        down_proj_24 = torch._C._nn.linear(
            mul_326,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_326 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_349 = hidden_states_345 + down_proj_24
        hidden_states_345 = down_proj_24 = None
        hidden_states_350 = hidden_states_349.to(torch.float32)
        pow_101 = hidden_states_350.pow(2)
        variance_100 = pow_101.mean(-1, keepdim=True)
        pow_101 = None
        add_200 = variance_100 + 1e-06
        variance_100 = None
        rsqrt_100 = torch.rsqrt(add_200)
        add_200 = None
        hidden_states_351 = hidden_states_350 * rsqrt_100
        hidden_states_350 = rsqrt_100 = None
        to_207 = hidden_states_351.to(torch.bfloat16)
        hidden_states_351 = None
        hidden_states_352 = (
            l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
            * to_207
        )
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            to_207
        ) = None
        linear_175 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_75 = linear_175.view((1, 20, -1, 128))
        linear_175 = None
        hidden_states_353 = view_75.to(torch.float32)
        view_75 = None
        pow_102 = hidden_states_353.pow(2)
        variance_101 = pow_102.mean(-1, keepdim=True)
        pow_102 = None
        add_201 = variance_101 + 1e-06
        variance_101 = None
        rsqrt_101 = torch.rsqrt(add_201)
        add_201 = None
        hidden_states_354 = hidden_states_353 * rsqrt_101
        hidden_states_353 = rsqrt_101 = None
        to_209 = hidden_states_354.to(torch.bfloat16)
        hidden_states_354 = None
        mul_330 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_
            * to_209
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_209
        ) = None
        query_states_25 = mul_330.transpose(1, 2)
        mul_330 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_76 = linear_176.view((1, 20, -1, 128))
        linear_176 = None
        hidden_states_355 = view_76.to(torch.float32)
        view_76 = None
        pow_103 = hidden_states_355.pow(2)
        variance_102 = pow_103.mean(-1, keepdim=True)
        pow_103 = None
        add_202 = variance_102 + 1e-06
        variance_102 = None
        rsqrt_102 = torch.rsqrt(add_202)
        add_202 = None
        hidden_states_356 = hidden_states_355 * rsqrt_102
        hidden_states_355 = rsqrt_102 = None
        to_211 = hidden_states_356.to(torch.bfloat16)
        hidden_states_356 = None
        mul_332 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_
            * to_211
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_211
        ) = None
        key_states_25 = mul_332.transpose(1, 2)
        mul_332 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_352 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_77 = linear_177.view((1, 20, -1, 128))
        linear_177 = None
        value_states_25 = view_77.transpose(1, 2)
        view_77 = None
        cos_28 = cos_2.unsqueeze(1)
        sin_28 = sin_2.unsqueeze(1)
        mul_333 = query_states_25 * cos_28
        x1_50 = query_states_25[(Ellipsis, slice(None, 64, None))]
        x2_50 = query_states_25[(Ellipsis, slice(64, None, None))]
        query_states_25 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_51 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_334 = cat_51 * sin_28
        cat_51 = None
        q_embed_25 = mul_333 + mul_334
        mul_333 = mul_334 = None
        mul_335 = key_states_25 * cos_28
        cos_28 = None
        x1_51 = key_states_25[(Ellipsis, slice(None, 64, None))]
        x2_51 = key_states_25[(Ellipsis, slice(64, None, None))]
        key_states_25 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_52 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_336 = cat_52 * sin_28
        cat_52 = sin_28 = None
        k_embed_25 = mul_335 + mul_336
        mul_335 = mul_336 = None
        getitem_181 = k_embed_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_25 = None
        hidden_states_357 = getitem_181.expand(1, 8, 2, 20, 128)
        getitem_181 = None
        key_25 = hidden_states_357.reshape(1, 16, 20, 128)
        hidden_states_357 = None
        getitem_182 = value_states_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_25 = None
        hidden_states_358 = getitem_182.expand(1, 8, 2, 20, 128)
        getitem_182 = None
        value_25 = hidden_states_358.reshape(1, 16, 20, 128)
        hidden_states_358 = None
        attention_mask_26 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            q_embed_25,
            key_25,
            value_25,
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_25 = key_25 = value_25 = attention_mask_26 = None
        transpose_104 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_104.contiguous()
        transpose_104 = None
        reshape_77 = attn_output_101.reshape(1, 20, -1)
        attn_output_101 = None
        attn_output_102 = reshape_77.contiguous()
        reshape_77 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_359 = hidden_states_349 + attn_output_103
        hidden_states_349 = attn_output_103 = None
        hidden_states_360 = hidden_states_359.to(torch.float32)
        pow_104 = hidden_states_360.pow(2)
        variance_103 = pow_104.mean(-1, keepdim=True)
        pow_104 = None
        add_206 = variance_103 + 1e-06
        variance_103 = None
        rsqrt_103 = torch.rsqrt(add_206)
        add_206 = None
        hidden_states_361 = hidden_states_360 * rsqrt_103
        hidden_states_360 = rsqrt_103 = None
        to_213 = hidden_states_361.to(torch.bfloat16)
        hidden_states_361 = None
        hidden_states_362 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
            * to_213
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            to_213
        ) = None
        linear_179 = torch._C._nn.linear(
            hidden_states_362,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_179, inplace=False)
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_362,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_362 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_339 = silu_25 * linear_180
        silu_25 = linear_180 = None
        down_proj_25 = torch._C._nn.linear(
            mul_339,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_339 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_363 = hidden_states_359 + down_proj_25
        hidden_states_359 = down_proj_25 = None
        hidden_states_364 = hidden_states_363.to(torch.float32)
        pow_105 = hidden_states_364.pow(2)
        variance_104 = pow_105.mean(-1, keepdim=True)
        pow_105 = None
        add_208 = variance_104 + 1e-06
        variance_104 = None
        rsqrt_104 = torch.rsqrt(add_208)
        add_208 = None
        hidden_states_365 = hidden_states_364 * rsqrt_104
        hidden_states_364 = rsqrt_104 = None
        to_215 = hidden_states_365.to(torch.bfloat16)
        hidden_states_365 = None
        hidden_states_366 = (
            l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
            * to_215
        )
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            to_215
        ) = None
        linear_182 = torch._C._nn.linear(
            hidden_states_366,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_78 = linear_182.view((1, 20, -1, 128))
        linear_182 = None
        hidden_states_367 = view_78.to(torch.float32)
        view_78 = None
        pow_106 = hidden_states_367.pow(2)
        variance_105 = pow_106.mean(-1, keepdim=True)
        pow_106 = None
        add_209 = variance_105 + 1e-06
        variance_105 = None
        rsqrt_105 = torch.rsqrt(add_209)
        add_209 = None
        hidden_states_368 = hidden_states_367 * rsqrt_105
        hidden_states_367 = rsqrt_105 = None
        to_217 = hidden_states_368.to(torch.bfloat16)
        hidden_states_368 = None
        mul_343 = (
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_
            * to_217
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_217
        ) = None
        query_states_26 = mul_343.transpose(1, 2)
        mul_343 = None
        linear_183 = torch._C._nn.linear(
            hidden_states_366,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_79 = linear_183.view((1, 20, -1, 128))
        linear_183 = None
        hidden_states_369 = view_79.to(torch.float32)
        view_79 = None
        pow_107 = hidden_states_369.pow(2)
        variance_106 = pow_107.mean(-1, keepdim=True)
        pow_107 = None
        add_210 = variance_106 + 1e-06
        variance_106 = None
        rsqrt_106 = torch.rsqrt(add_210)
        add_210 = None
        hidden_states_370 = hidden_states_369 * rsqrt_106
        hidden_states_369 = rsqrt_106 = None
        to_219 = hidden_states_370.to(torch.bfloat16)
        hidden_states_370 = None
        mul_345 = (
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_
            * to_219
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_219
        ) = None
        key_states_26 = mul_345.transpose(1, 2)
        mul_345 = None
        linear_184 = torch._C._nn.linear(
            hidden_states_366,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_366 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_80 = linear_184.view((1, 20, -1, 128))
        linear_184 = None
        value_states_26 = view_80.transpose(1, 2)
        view_80 = None
        cos_29 = cos_2.unsqueeze(1)
        sin_29 = sin_2.unsqueeze(1)
        mul_346 = query_states_26 * cos_29
        x1_52 = query_states_26[(Ellipsis, slice(None, 64, None))]
        x2_52 = query_states_26[(Ellipsis, slice(64, None, None))]
        query_states_26 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_53 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_347 = cat_53 * sin_29
        cat_53 = None
        q_embed_26 = mul_346 + mul_347
        mul_346 = mul_347 = None
        mul_348 = key_states_26 * cos_29
        cos_29 = None
        x1_53 = key_states_26[(Ellipsis, slice(None, 64, None))]
        x2_53 = key_states_26[(Ellipsis, slice(64, None, None))]
        key_states_26 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_54 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_349 = cat_54 * sin_29
        cat_54 = sin_29 = None
        k_embed_26 = mul_348 + mul_349
        mul_348 = mul_349 = None
        getitem_188 = k_embed_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_26 = None
        hidden_states_371 = getitem_188.expand(1, 8, 2, 20, 128)
        getitem_188 = None
        key_26 = hidden_states_371.reshape(1, 16, 20, 128)
        hidden_states_371 = None
        getitem_189 = value_states_26[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_26 = None
        hidden_states_372 = getitem_189.expand(1, 8, 2, 20, 128)
        getitem_189 = None
        value_26 = hidden_states_372.reshape(1, 16, 20, 128)
        hidden_states_372 = None
        attention_mask_27 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            q_embed_26,
            key_26,
            value_26,
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_26 = key_26 = value_26 = attention_mask_27 = None
        transpose_108 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_108.contiguous()
        transpose_108 = None
        reshape_80 = attn_output_105.reshape(1, 20, -1)
        attn_output_105 = None
        attn_output_106 = reshape_80.contiguous()
        reshape_80 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_373 = hidden_states_363 + attn_output_107
        hidden_states_363 = attn_output_107 = None
        hidden_states_374 = hidden_states_373.to(torch.float32)
        pow_108 = hidden_states_374.pow(2)
        variance_107 = pow_108.mean(-1, keepdim=True)
        pow_108 = None
        add_214 = variance_107 + 1e-06
        variance_107 = None
        rsqrt_107 = torch.rsqrt(add_214)
        add_214 = None
        hidden_states_375 = hidden_states_374 * rsqrt_107
        hidden_states_374 = rsqrt_107 = None
        to_221 = hidden_states_375.to(torch.bfloat16)
        hidden_states_375 = None
        hidden_states_376 = (
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
            * to_221
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = (
            to_221
        ) = None
        linear_186 = torch._C._nn.linear(
            hidden_states_376,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_186, inplace=False)
        linear_186 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_376,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_376 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_352 = silu_26 * linear_187
        silu_26 = linear_187 = None
        down_proj_26 = torch._C._nn.linear(
            mul_352,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_352 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_377 = hidden_states_373 + down_proj_26
        hidden_states_373 = down_proj_26 = None
        hidden_states_378 = hidden_states_377.to(torch.float32)
        pow_109 = hidden_states_378.pow(2)
        variance_108 = pow_109.mean(-1, keepdim=True)
        pow_109 = None
        add_216 = variance_108 + 1e-06
        variance_108 = None
        rsqrt_108 = torch.rsqrt(add_216)
        add_216 = None
        hidden_states_379 = hidden_states_378 * rsqrt_108
        hidden_states_378 = rsqrt_108 = None
        to_223 = hidden_states_379.to(torch.bfloat16)
        hidden_states_379 = None
        hidden_states_380 = (
            l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
            * to_223
        )
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            to_223
        ) = None
        linear_189 = torch._C._nn.linear(
            hidden_states_380,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_81 = linear_189.view((1, 20, -1, 128))
        linear_189 = None
        hidden_states_381 = view_81.to(torch.float32)
        view_81 = None
        pow_110 = hidden_states_381.pow(2)
        variance_109 = pow_110.mean(-1, keepdim=True)
        pow_110 = None
        add_217 = variance_109 + 1e-06
        variance_109 = None
        rsqrt_109 = torch.rsqrt(add_217)
        add_217 = None
        hidden_states_382 = hidden_states_381 * rsqrt_109
        hidden_states_381 = rsqrt_109 = None
        to_225 = hidden_states_382.to(torch.bfloat16)
        hidden_states_382 = None
        mul_356 = (
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_
            * to_225
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_225
        ) = None
        query_states_27 = mul_356.transpose(1, 2)
        mul_356 = None
        linear_190 = torch._C._nn.linear(
            hidden_states_380,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_82 = linear_190.view((1, 20, -1, 128))
        linear_190 = None
        hidden_states_383 = view_82.to(torch.float32)
        view_82 = None
        pow_111 = hidden_states_383.pow(2)
        variance_110 = pow_111.mean(-1, keepdim=True)
        pow_111 = None
        add_218 = variance_110 + 1e-06
        variance_110 = None
        rsqrt_110 = torch.rsqrt(add_218)
        add_218 = None
        hidden_states_384 = hidden_states_383 * rsqrt_110
        hidden_states_383 = rsqrt_110 = None
        to_227 = hidden_states_384.to(torch.bfloat16)
        hidden_states_384 = None
        mul_358 = (
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_
            * to_227
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_227
        ) = None
        key_states_27 = mul_358.transpose(1, 2)
        mul_358 = None
        linear_191 = torch._C._nn.linear(
            hidden_states_380,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_380 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_83 = linear_191.view((1, 20, -1, 128))
        linear_191 = None
        value_states_27 = view_83.transpose(1, 2)
        view_83 = None
        cos_30 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_30 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_359 = query_states_27 * cos_30
        x1_54 = query_states_27[(Ellipsis, slice(None, 64, None))]
        x2_54 = query_states_27[(Ellipsis, slice(64, None, None))]
        query_states_27 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_55 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_360 = cat_55 * sin_30
        cat_55 = None
        q_embed_27 = mul_359 + mul_360
        mul_359 = mul_360 = None
        mul_361 = key_states_27 * cos_30
        cos_30 = None
        x1_55 = key_states_27[(Ellipsis, slice(None, 64, None))]
        x2_55 = key_states_27[(Ellipsis, slice(64, None, None))]
        key_states_27 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_56 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_362 = cat_56 * sin_30
        cat_56 = sin_30 = None
        k_embed_27 = mul_361 + mul_362
        mul_361 = mul_362 = None
        getitem_195 = k_embed_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_27 = None
        hidden_states_385 = getitem_195.expand(1, 8, 2, 20, 128)
        getitem_195 = None
        key_27 = hidden_states_385.reshape(1, 16, 20, 128)
        hidden_states_385 = None
        getitem_196 = value_states_27[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_27 = None
        hidden_states_386 = getitem_196.expand(1, 8, 2, 20, 128)
        getitem_196 = None
        value_27 = hidden_states_386.reshape(1, 16, 20, 128)
        hidden_states_386 = None
        attention_mask_28 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        causal_mask = None
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            q_embed_27,
            key_27,
            value_27,
            attn_mask=attention_mask_28,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        q_embed_27 = key_27 = value_27 = attention_mask_28 = None
        transpose_112 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_112.contiguous()
        transpose_112 = None
        reshape_83 = attn_output_109.reshape(1, 20, -1)
        attn_output_109 = None
        attn_output_110 = reshape_83.contiguous()
        reshape_83 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_387 = hidden_states_377 + attn_output_111
        hidden_states_377 = attn_output_111 = None
        hidden_states_388 = hidden_states_387.to(torch.float32)
        pow_112 = hidden_states_388.pow(2)
        variance_111 = pow_112.mean(-1, keepdim=True)
        pow_112 = None
        add_222 = variance_111 + 1e-06
        variance_111 = None
        rsqrt_111 = torch.rsqrt(add_222)
        add_222 = None
        hidden_states_389 = hidden_states_388 * rsqrt_111
        hidden_states_388 = rsqrt_111 = None
        to_229 = hidden_states_389.to(torch.bfloat16)
        hidden_states_389 = None
        hidden_states_390 = (
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
            * to_229
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = (
            to_229
        ) = None
        linear_193 = torch._C._nn.linear(
            hidden_states_390,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_193, inplace=False)
        linear_193 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_390,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_390 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_365 = silu_27 * linear_194
        silu_27 = linear_194 = None
        down_proj_27 = torch._C._nn.linear(
            mul_365,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_365 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_391 = hidden_states_387 + down_proj_27
        hidden_states_387 = down_proj_27 = None
        hidden_states_392 = hidden_states_391.to(torch.float32)
        hidden_states_391 = None
        pow_113 = hidden_states_392.pow(2)
        variance_112 = pow_113.mean(-1, keepdim=True)
        pow_113 = None
        add_224 = variance_112 + 1e-06
        variance_112 = None
        rsqrt_112 = torch.rsqrt(add_224)
        add_224 = None
        hidden_states_393 = hidden_states_392 * rsqrt_112
        hidden_states_392 = rsqrt_112 = None
        to_231 = hidden_states_393.to(torch.bfloat16)
        hidden_states_393 = None
        hidden_states_394 = (
            l_self_modules_model_modules_norm_parameters_weight_ * to_231
        )
        l_self_modules_model_modules_norm_parameters_weight_ = to_231 = None
        getitem_198 = hidden_states_394[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_394 = None
        logits = torch._C._nn.linear(
            getitem_198,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
        )
        getitem_198 = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        return (logits,)
