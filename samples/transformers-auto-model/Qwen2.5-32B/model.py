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
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_lm_head_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_36_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_37_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_38_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_39_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_40_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_41_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_42_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_43_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_44_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_45_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_46_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_47_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_48_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_49_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_50_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_51_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_52_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_53_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_54_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_55_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_56_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_57_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_58_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_59_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_60_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_61_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_62_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_bias_
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_63_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
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
        l_kwargs_input_ids_ = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        cache_position = torch.arange(0, 19, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange = torch.arange(19, device=device(type="cpu"))
        kv_arange += 0
        kv_arange_1 = kv_arange
        kv_arange = None
        batch_arange = torch.arange(1, device=device(type="cpu"))
        head_arange = torch.arange(1, device=device(type="cpu"))
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
            19, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            19, "error"
        )
        _vmap_increment_nesting_3 = None
        child_3 = torch._C._functorch._add_batch_dim(kv_arange_1, 0, 4)
        kv_arange_1 = None
        result = child_2.new_ones((), dtype=torch.bool)
        le = child_3.le(child_2)
        child_2 = None
        result_1 = result.__and__(le)
        result = le = None
        function_ctx = torch.autograd.function.FunctionCtx()
        function_ctx = None
        index = torch.ops.aten.index(attention_mask, [child, child_3])
        attention_mask = child = child_3 = None
        result_2 = result_1.__and__(index)
        result_1 = index = None
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 19, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 19, 0
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
        getitem = l_self_modules_model_modules_rotary_emb_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_model_modules_rotary_emb_buffers_inv_freq_ = None
        float_1 = getitem.float()
        getitem = None
        expand = float_1.expand(1, -1, 1)
        float_1 = None
        inv_freq_expanded = expand.to(device(type="cpu"))
        expand = None
        getitem_1 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded = getitem_1.float()
        getitem_1 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast(
            "cpu", None, False, None
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
        cos_2 = cos_1.to(dtype=torch.float16)
        cos_1 = None
        sin_2 = sin_1.to(dtype=torch.float16)
        sin_1 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        hidden_states = inputs_embeds.to(torch.float32)
        pow_1 = hidden_states.pow(2)
        variance = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = variance + 1e-05
        variance = None
        rsqrt = torch.rsqrt(add)
        add = None
        hidden_states_1 = hidden_states * rsqrt
        hidden_states = rsqrt = None
        to_5 = hidden_states_1.to(torch.float16)
        hidden_states_1 = None
        hidden_states_2 = (
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
            * to_5
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            to_5
        ) = None
        linear = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view = linear.view((1, 19, -1, 128))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_1 = linear_1.view((1, 19, -1, 128))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_2 = linear_2.view((1, 19, -1, 128))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_4 = query_states * cos_3
        x1 = query_states[(Ellipsis, slice(None, 64, None))]
        x2 = query_states[(Ellipsis, slice(64, None, None))]
        query_states = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_5 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_4 + mul_5
        mul_4 = mul_5 = None
        mul_6 = key_states * cos_3
        cos_3 = None
        x1_1 = key_states[(Ellipsis, slice(None, 64, None))]
        x2_1 = key_states[(Ellipsis, slice(64, None, None))]
        key_states = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_7 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_6 + mul_7
        mul_6 = mul_7 = None
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
        hidden_states_3 = getitem_6.expand(1, 8, 5, 19, 128)
        getitem_6 = None
        key = hidden_states_3.reshape(1, 40, 19, 128)
        hidden_states_3 = None
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
        hidden_states_4 = getitem_7.expand(1, 8, 5, 19, 128)
        getitem_7 = None
        value = hidden_states_4.reshape(1, 40, 19, 128)
        hidden_states_4 = None
        attention_mask_1 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_2 = attn_output_1.reshape(1, 19, -1)
        attn_output_1 = None
        attn_output_2 = reshape_2.contiguous()
        reshape_2 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_5 = inputs_embeds + attn_output_3
        inputs_embeds = attn_output_3 = None
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
        to_7 = hidden_states_7.to(torch.float16)
        hidden_states_7 = None
        hidden_states_8 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_7
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_7
        ) = None
        linear_4 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_8 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_10 = silu * linear_5
        silu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_10,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_10 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_9 = hidden_states_11.to(torch.float16)
        hidden_states_11 = None
        hidden_states_12 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            to_9
        ) = None
        linear_7 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_3 = linear_7.view((1, 19, -1, 128))
        linear_7 = None
        query_states_1 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_4 = linear_8.view((1, 19, -1, 128))
        linear_8 = None
        key_states_1 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_12 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_5 = linear_9.view((1, 19, -1, 128))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_13 = query_states_1 * cos_4
        x1_2 = query_states_1[(Ellipsis, slice(None, 64, None))]
        x2_2 = query_states_1[(Ellipsis, slice(64, None, None))]
        query_states_1 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_14 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_13 + mul_14
        mul_13 = mul_14 = None
        mul_15 = key_states_1 * cos_4
        cos_4 = None
        x1_3 = key_states_1[(Ellipsis, slice(None, 64, None))]
        x2_3 = key_states_1[(Ellipsis, slice(64, None, None))]
        key_states_1 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_16 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_15 + mul_16
        mul_15 = mul_16 = None
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
        hidden_states_13 = getitem_13.expand(1, 8, 5, 19, 128)
        getitem_13 = None
        key_2 = hidden_states_13.reshape(1, 40, 19, 128)
        hidden_states_13 = None
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
        hidden_states_14 = getitem_14.expand(1, 8, 5, 19, 128)
        getitem_14 = None
        value_2 = hidden_states_14.reshape(1, 40, 19, 128)
        hidden_states_14 = None
        attention_mask_2 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_5 = attn_output_5.reshape(1, 19, -1)
        attn_output_5 = None
        attn_output_6 = reshape_5.contiguous()
        reshape_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_11 = hidden_states_17.to(torch.float16)
        hidden_states_17 = None
        hidden_states_18 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_11
        ) = None
        linear_11 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_18,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_18 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_19 = silu_1 * linear_12
        silu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_19,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_19 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_13 = hidden_states_21.to(torch.float16)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
            * to_13
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            to_13
        ) = None
        linear_14 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_6 = linear_14.view((1, 19, -1, 128))
        linear_14 = None
        query_states_2 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_7 = linear_15.view((1, 19, -1, 128))
        linear_15 = None
        key_states_2 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_22 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_8 = linear_16.view((1, 19, -1, 128))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_22 = query_states_2 * cos_5
        x1_4 = query_states_2[(Ellipsis, slice(None, 64, None))]
        x2_4 = query_states_2[(Ellipsis, slice(64, None, None))]
        query_states_2 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_23 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_22 + mul_23
        mul_22 = mul_23 = None
        mul_24 = key_states_2 * cos_5
        cos_5 = None
        x1_5 = key_states_2[(Ellipsis, slice(None, 64, None))]
        x2_5 = key_states_2[(Ellipsis, slice(64, None, None))]
        key_states_2 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_25 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_24 + mul_25
        mul_24 = mul_25 = None
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
        hidden_states_23 = getitem_20.expand(1, 8, 5, 19, 128)
        getitem_20 = None
        key_4 = hidden_states_23.reshape(1, 40, 19, 128)
        hidden_states_23 = None
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
        hidden_states_24 = getitem_21.expand(1, 8, 5, 19, 128)
        getitem_21 = None
        value_4 = hidden_states_24.reshape(1, 40, 19, 128)
        hidden_states_24 = None
        attention_mask_3 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_8 = attn_output_9.reshape(1, 19, -1)
        attn_output_9 = None
        attn_output_10 = reshape_8.contiguous()
        reshape_8 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_15 = hidden_states_27.to(torch.float16)
        hidden_states_27 = None
        hidden_states_28 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_15
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_15
        ) = None
        linear_18 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_28,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_28 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_28 = silu_2 * linear_19
        silu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_28,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_28 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_17 = hidden_states_31.to(torch.float16)
        hidden_states_31 = None
        hidden_states_32 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            to_17
        ) = None
        linear_21 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_9 = linear_21.view((1, 19, -1, 128))
        linear_21 = None
        query_states_3 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_10 = linear_22.view((1, 19, -1, 128))
        linear_22 = None
        key_states_3 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_32 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_11 = linear_23.view((1, 19, -1, 128))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_31 = query_states_3 * cos_6
        x1_6 = query_states_3[(Ellipsis, slice(None, 64, None))]
        x2_6 = query_states_3[(Ellipsis, slice(64, None, None))]
        query_states_3 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_32 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_31 + mul_32
        mul_31 = mul_32 = None
        mul_33 = key_states_3 * cos_6
        cos_6 = None
        x1_7 = key_states_3[(Ellipsis, slice(None, 64, None))]
        x2_7 = key_states_3[(Ellipsis, slice(64, None, None))]
        key_states_3 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_34 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_33 + mul_34
        mul_33 = mul_34 = None
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
        hidden_states_33 = getitem_27.expand(1, 8, 5, 19, 128)
        getitem_27 = None
        key_6 = hidden_states_33.reshape(1, 40, 19, 128)
        hidden_states_33 = None
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
        hidden_states_34 = getitem_28.expand(1, 8, 5, 19, 128)
        getitem_28 = None
        value_6 = hidden_states_34.reshape(1, 40, 19, 128)
        hidden_states_34 = None
        attention_mask_4 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_11 = attn_output_13.reshape(1, 19, -1)
        attn_output_13 = None
        attn_output_14 = reshape_11.contiguous()
        reshape_11 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_19 = hidden_states_37.to(torch.float16)
        hidden_states_37 = None
        hidden_states_38 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_19
        ) = None
        linear_25 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_38,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_38 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_37 = silu_3 * linear_26
        silu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_37,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_37 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_21 = hidden_states_41.to(torch.float16)
        hidden_states_41 = None
        hidden_states_42 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
            * to_21
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            to_21
        ) = None
        linear_28 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_12 = linear_28.view((1, 19, -1, 128))
        linear_28 = None
        query_states_4 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_13 = linear_29.view((1, 19, -1, 128))
        linear_29 = None
        key_states_4 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_42 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_14 = linear_30.view((1, 19, -1, 128))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_40 = query_states_4 * cos_7
        x1_8 = query_states_4[(Ellipsis, slice(None, 64, None))]
        x2_8 = query_states_4[(Ellipsis, slice(64, None, None))]
        query_states_4 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_41 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_40 + mul_41
        mul_40 = mul_41 = None
        mul_42 = key_states_4 * cos_7
        cos_7 = None
        x1_9 = key_states_4[(Ellipsis, slice(None, 64, None))]
        x2_9 = key_states_4[(Ellipsis, slice(64, None, None))]
        key_states_4 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_43 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_42 + mul_43
        mul_42 = mul_43 = None
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
        hidden_states_43 = getitem_34.expand(1, 8, 5, 19, 128)
        getitem_34 = None
        key_8 = hidden_states_43.reshape(1, 40, 19, 128)
        hidden_states_43 = None
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
        hidden_states_44 = getitem_35.expand(1, 8, 5, 19, 128)
        getitem_35 = None
        value_8 = hidden_states_44.reshape(1, 40, 19, 128)
        hidden_states_44 = None
        attention_mask_5 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_14 = attn_output_17.reshape(1, 19, -1)
        attn_output_17 = None
        attn_output_18 = reshape_14.contiguous()
        reshape_14 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_23 = hidden_states_47.to(torch.float16)
        hidden_states_47 = None
        hidden_states_48 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_23
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_23
        ) = None
        linear_32 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_48 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_46 = silu_4 * linear_33
        silu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_46,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_46 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_25 = hidden_states_51.to(torch.float16)
        hidden_states_51 = None
        hidden_states_52 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            to_25
        ) = None
        linear_35 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_15 = linear_35.view((1, 19, -1, 128))
        linear_35 = None
        query_states_5 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_16 = linear_36.view((1, 19, -1, 128))
        linear_36 = None
        key_states_5 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_52 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_17 = linear_37.view((1, 19, -1, 128))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_49 = query_states_5 * cos_8
        x1_10 = query_states_5[(Ellipsis, slice(None, 64, None))]
        x2_10 = query_states_5[(Ellipsis, slice(64, None, None))]
        query_states_5 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_50 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_49 + mul_50
        mul_49 = mul_50 = None
        mul_51 = key_states_5 * cos_8
        cos_8 = None
        x1_11 = key_states_5[(Ellipsis, slice(None, 64, None))]
        x2_11 = key_states_5[(Ellipsis, slice(64, None, None))]
        key_states_5 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_52 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_51 + mul_52
        mul_51 = mul_52 = None
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
        hidden_states_53 = getitem_41.expand(1, 8, 5, 19, 128)
        getitem_41 = None
        key_10 = hidden_states_53.reshape(1, 40, 19, 128)
        hidden_states_53 = None
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
        hidden_states_54 = getitem_42.expand(1, 8, 5, 19, 128)
        getitem_42 = None
        value_10 = hidden_states_54.reshape(1, 40, 19, 128)
        hidden_states_54 = None
        attention_mask_6 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_17 = attn_output_21.reshape(1, 19, -1)
        attn_output_21 = None
        attn_output_22 = reshape_17.contiguous()
        reshape_17 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_27 = hidden_states_57.to(torch.float16)
        hidden_states_57 = None
        hidden_states_58 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_27
        ) = None
        linear_39 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_58,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_58 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_55 = silu_5 * linear_40
        silu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_55,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_55 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_29 = hidden_states_61.to(torch.float16)
        hidden_states_61 = None
        hidden_states_62 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
            * to_29
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            to_29
        ) = None
        linear_42 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_18 = linear_42.view((1, 19, -1, 128))
        linear_42 = None
        query_states_6 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_19 = linear_43.view((1, 19, -1, 128))
        linear_43 = None
        key_states_6 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_62 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_20 = linear_44.view((1, 19, -1, 128))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_58 = query_states_6 * cos_9
        x1_12 = query_states_6[(Ellipsis, slice(None, 64, None))]
        x2_12 = query_states_6[(Ellipsis, slice(64, None, None))]
        query_states_6 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_59 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_58 + mul_59
        mul_58 = mul_59 = None
        mul_60 = key_states_6 * cos_9
        cos_9 = None
        x1_13 = key_states_6[(Ellipsis, slice(None, 64, None))]
        x2_13 = key_states_6[(Ellipsis, slice(64, None, None))]
        key_states_6 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_61 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_60 + mul_61
        mul_60 = mul_61 = None
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
        hidden_states_63 = getitem_48.expand(1, 8, 5, 19, 128)
        getitem_48 = None
        key_12 = hidden_states_63.reshape(1, 40, 19, 128)
        hidden_states_63 = None
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
        hidden_states_64 = getitem_49.expand(1, 8, 5, 19, 128)
        getitem_49 = None
        value_12 = hidden_states_64.reshape(1, 40, 19, 128)
        hidden_states_64 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_20 = attn_output_25.reshape(1, 19, -1)
        attn_output_25 = None
        attn_output_26 = reshape_20.contiguous()
        reshape_20 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_31 = hidden_states_67.to(torch.float16)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_31
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_31
        ) = None
        linear_46 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_68,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_68 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_64 = silu_6 * linear_47
        silu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_64,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_64 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_33 = hidden_states_71.to(torch.float16)
        hidden_states_71 = None
        hidden_states_72 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            to_33
        ) = None
        linear_49 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_21 = linear_49.view((1, 19, -1, 128))
        linear_49 = None
        query_states_7 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_22 = linear_50.view((1, 19, -1, 128))
        linear_50 = None
        key_states_7 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_72 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_23 = linear_51.view((1, 19, -1, 128))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_67 = query_states_7 * cos_10
        x1_14 = query_states_7[(Ellipsis, slice(None, 64, None))]
        x2_14 = query_states_7[(Ellipsis, slice(64, None, None))]
        query_states_7 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_68 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_67 + mul_68
        mul_67 = mul_68 = None
        mul_69 = key_states_7 * cos_10
        cos_10 = None
        x1_15 = key_states_7[(Ellipsis, slice(None, 64, None))]
        x2_15 = key_states_7[(Ellipsis, slice(64, None, None))]
        key_states_7 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_70 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_69 + mul_70
        mul_69 = mul_70 = None
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
        hidden_states_73 = getitem_55.expand(1, 8, 5, 19, 128)
        getitem_55 = None
        key_14 = hidden_states_73.reshape(1, 40, 19, 128)
        hidden_states_73 = None
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
        hidden_states_74 = getitem_56.expand(1, 8, 5, 19, 128)
        getitem_56 = None
        value_14 = hidden_states_74.reshape(1, 40, 19, 128)
        hidden_states_74 = None
        attention_mask_8 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_23 = attn_output_29.reshape(1, 19, -1)
        attn_output_29 = None
        attn_output_30 = reshape_23.contiguous()
        reshape_23 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_35 = hidden_states_77.to(torch.float16)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_35
        ) = None
        linear_53 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_78,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_78 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_73 = silu_7 * linear_54
        silu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_73,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_73 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_37 = hidden_states_81.to(torch.float16)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
            * to_37
        )
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            to_37
        ) = None
        linear_56 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_24 = linear_56.view((1, 19, -1, 128))
        linear_56 = None
        query_states_8 = view_24.transpose(1, 2)
        view_24 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_25 = linear_57.view((1, 19, -1, 128))
        linear_57 = None
        key_states_8 = view_25.transpose(1, 2)
        view_25 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_82 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_26 = linear_58.view((1, 19, -1, 128))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_76 = query_states_8 * cos_11
        x1_16 = query_states_8[(Ellipsis, slice(None, 64, None))]
        x2_16 = query_states_8[(Ellipsis, slice(64, None, None))]
        query_states_8 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_77 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_76 + mul_77
        mul_76 = mul_77 = None
        mul_78 = key_states_8 * cos_11
        cos_11 = None
        x1_17 = key_states_8[(Ellipsis, slice(None, 64, None))]
        x2_17 = key_states_8[(Ellipsis, slice(64, None, None))]
        key_states_8 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_79 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_78 + mul_79
        mul_78 = mul_79 = None
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
        hidden_states_83 = getitem_62.expand(1, 8, 5, 19, 128)
        getitem_62 = None
        key_16 = hidden_states_83.reshape(1, 40, 19, 128)
        hidden_states_83 = None
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
        hidden_states_84 = getitem_63.expand(1, 8, 5, 19, 128)
        getitem_63 = None
        value_16 = hidden_states_84.reshape(1, 40, 19, 128)
        hidden_states_84 = None
        attention_mask_9 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_26 = attn_output_33.reshape(1, 19, -1)
        attn_output_33 = None
        attn_output_34 = reshape_26.contiguous()
        reshape_26 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_39 = hidden_states_87.to(torch.float16)
        hidden_states_87 = None
        hidden_states_88 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_39
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_39
        ) = None
        linear_60 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_60, inplace=False)
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_88 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_82 = silu_8 * linear_61
        silu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_82,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_82 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_41 = hidden_states_91.to(torch.float16)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
            * to_41
        )
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            to_41
        ) = None
        linear_63 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_27 = linear_63.view((1, 19, -1, 128))
        linear_63 = None
        query_states_9 = view_27.transpose(1, 2)
        view_27 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_28 = linear_64.view((1, 19, -1, 128))
        linear_64 = None
        key_states_9 = view_28.transpose(1, 2)
        view_28 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_92 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_29 = linear_65.view((1, 19, -1, 128))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_85 = query_states_9 * cos_12
        x1_18 = query_states_9[(Ellipsis, slice(None, 64, None))]
        x2_18 = query_states_9[(Ellipsis, slice(64, None, None))]
        query_states_9 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_86 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_85 + mul_86
        mul_85 = mul_86 = None
        mul_87 = key_states_9 * cos_12
        cos_12 = None
        x1_19 = key_states_9[(Ellipsis, slice(None, 64, None))]
        x2_19 = key_states_9[(Ellipsis, slice(64, None, None))]
        key_states_9 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_88 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_87 + mul_88
        mul_87 = mul_88 = None
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
        hidden_states_93 = getitem_69.expand(1, 8, 5, 19, 128)
        getitem_69 = None
        key_18 = hidden_states_93.reshape(1, 40, 19, 128)
        hidden_states_93 = None
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
        hidden_states_94 = getitem_70.expand(1, 8, 5, 19, 128)
        getitem_70 = None
        value_18 = hidden_states_94.reshape(1, 40, 19, 128)
        hidden_states_94 = None
        attention_mask_10 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_29 = attn_output_37.reshape(1, 19, -1)
        attn_output_37 = None
        attn_output_38 = reshape_29.contiguous()
        reshape_29 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_43 = hidden_states_97.to(torch.float16)
        hidden_states_97 = None
        hidden_states_98 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_43
        ) = None
        linear_67 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_67, inplace=False)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_98,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_98 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_91 = silu_9 * linear_68
        silu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_91,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_91 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_45 = hidden_states_101.to(torch.float16)
        hidden_states_101 = None
        hidden_states_102 = (
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
            * to_45
        )
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            to_45
        ) = None
        linear_70 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_30 = linear_70.view((1, 19, -1, 128))
        linear_70 = None
        query_states_10 = view_30.transpose(1, 2)
        view_30 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_31 = linear_71.view((1, 19, -1, 128))
        linear_71 = None
        key_states_10 = view_31.transpose(1, 2)
        view_31 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_102 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_32 = linear_72.view((1, 19, -1, 128))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_94 = query_states_10 * cos_13
        x1_20 = query_states_10[(Ellipsis, slice(None, 64, None))]
        x2_20 = query_states_10[(Ellipsis, slice(64, None, None))]
        query_states_10 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_95 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_94 + mul_95
        mul_94 = mul_95 = None
        mul_96 = key_states_10 * cos_13
        cos_13 = None
        x1_21 = key_states_10[(Ellipsis, slice(None, 64, None))]
        x2_21 = key_states_10[(Ellipsis, slice(64, None, None))]
        key_states_10 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_97 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_96 + mul_97
        mul_96 = mul_97 = None
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
        hidden_states_103 = getitem_76.expand(1, 8, 5, 19, 128)
        getitem_76 = None
        key_20 = hidden_states_103.reshape(1, 40, 19, 128)
        hidden_states_103 = None
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
        hidden_states_104 = getitem_77.expand(1, 8, 5, 19, 128)
        getitem_77 = None
        value_20 = hidden_states_104.reshape(1, 40, 19, 128)
        hidden_states_104 = None
        attention_mask_11 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_32 = attn_output_41.reshape(1, 19, -1)
        attn_output_41 = None
        attn_output_42 = reshape_32.contiguous()
        reshape_32 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_47 = hidden_states_107.to(torch.float16)
        hidden_states_107 = None
        hidden_states_108 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_47
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_47
        ) = None
        linear_74 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_108,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_108 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_100 = silu_10 * linear_75
        silu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_100,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_100 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_49 = hidden_states_111.to(torch.float16)
        hidden_states_111 = None
        hidden_states_112 = (
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            to_49
        ) = None
        linear_77 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_33 = linear_77.view((1, 19, -1, 128))
        linear_77 = None
        query_states_11 = view_33.transpose(1, 2)
        view_33 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_34 = linear_78.view((1, 19, -1, 128))
        linear_78 = None
        key_states_11 = view_34.transpose(1, 2)
        view_34 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_112 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_35 = linear_79.view((1, 19, -1, 128))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_103 = query_states_11 * cos_14
        x1_22 = query_states_11[(Ellipsis, slice(None, 64, None))]
        x2_22 = query_states_11[(Ellipsis, slice(64, None, None))]
        query_states_11 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_104 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_103 + mul_104
        mul_103 = mul_104 = None
        mul_105 = key_states_11 * cos_14
        cos_14 = None
        x1_23 = key_states_11[(Ellipsis, slice(None, 64, None))]
        x2_23 = key_states_11[(Ellipsis, slice(64, None, None))]
        key_states_11 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_106 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_105 + mul_106
        mul_105 = mul_106 = None
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
        hidden_states_113 = getitem_83.expand(1, 8, 5, 19, 128)
        getitem_83 = None
        key_22 = hidden_states_113.reshape(1, 40, 19, 128)
        hidden_states_113 = None
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
        hidden_states_114 = getitem_84.expand(1, 8, 5, 19, 128)
        getitem_84 = None
        value_22 = hidden_states_114.reshape(1, 40, 19, 128)
        hidden_states_114 = None
        attention_mask_12 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_35 = attn_output_45.reshape(1, 19, -1)
        attn_output_45 = None
        attn_output_46 = reshape_35.contiguous()
        reshape_35 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_51 = hidden_states_117.to(torch.float16)
        hidden_states_117 = None
        hidden_states_118 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_51
        ) = None
        linear_81 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_81, inplace=False)
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_118,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_118 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_109 = silu_11 * linear_82
        silu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_109,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_109 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_53 = hidden_states_121.to(torch.float16)
        hidden_states_121 = None
        hidden_states_122 = (
            l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
            * to_53
        )
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            to_53
        ) = None
        linear_84 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_36 = linear_84.view((1, 19, -1, 128))
        linear_84 = None
        query_states_12 = view_36.transpose(1, 2)
        view_36 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_37 = linear_85.view((1, 19, -1, 128))
        linear_85 = None
        key_states_12 = view_37.transpose(1, 2)
        view_37 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_122 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_38 = linear_86.view((1, 19, -1, 128))
        linear_86 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_112 = query_states_12 * cos_15
        x1_24 = query_states_12[(Ellipsis, slice(None, 64, None))]
        x2_24 = query_states_12[(Ellipsis, slice(64, None, None))]
        query_states_12 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_113 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_112 + mul_113
        mul_112 = mul_113 = None
        mul_114 = key_states_12 * cos_15
        cos_15 = None
        x1_25 = key_states_12[(Ellipsis, slice(None, 64, None))]
        x2_25 = key_states_12[(Ellipsis, slice(64, None, None))]
        key_states_12 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_115 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_114 + mul_115
        mul_114 = mul_115 = None
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
        hidden_states_123 = getitem_90.expand(1, 8, 5, 19, 128)
        getitem_90 = None
        key_24 = hidden_states_123.reshape(1, 40, 19, 128)
        hidden_states_123 = None
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
        hidden_states_124 = getitem_91.expand(1, 8, 5, 19, 128)
        getitem_91 = None
        value_24 = hidden_states_124.reshape(1, 40, 19, 128)
        hidden_states_124 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_38 = attn_output_49.reshape(1, 19, -1)
        attn_output_49 = None
        attn_output_50 = reshape_38.contiguous()
        reshape_38 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_55 = hidden_states_127.to(torch.float16)
        hidden_states_127 = None
        hidden_states_128 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_55
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_55
        ) = None
        linear_88 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_88, inplace=False)
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_128 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_118 = silu_12 * linear_89
        silu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_118,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_118 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_57 = hidden_states_131.to(torch.float16)
        hidden_states_131 = None
        hidden_states_132 = (
            l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
            * to_57
        )
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            to_57
        ) = None
        linear_91 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_39 = linear_91.view((1, 19, -1, 128))
        linear_91 = None
        query_states_13 = view_39.transpose(1, 2)
        view_39 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_40 = linear_92.view((1, 19, -1, 128))
        linear_92 = None
        key_states_13 = view_40.transpose(1, 2)
        view_40 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_132 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_41 = linear_93.view((1, 19, -1, 128))
        linear_93 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_121 = query_states_13 * cos_16
        x1_26 = query_states_13[(Ellipsis, slice(None, 64, None))]
        x2_26 = query_states_13[(Ellipsis, slice(64, None, None))]
        query_states_13 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_122 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_121 + mul_122
        mul_121 = mul_122 = None
        mul_123 = key_states_13 * cos_16
        cos_16 = None
        x1_27 = key_states_13[(Ellipsis, slice(None, 64, None))]
        x2_27 = key_states_13[(Ellipsis, slice(64, None, None))]
        key_states_13 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_124 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_123 + mul_124
        mul_123 = mul_124 = None
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
        hidden_states_133 = getitem_97.expand(1, 8, 5, 19, 128)
        getitem_97 = None
        key_26 = hidden_states_133.reshape(1, 40, 19, 128)
        hidden_states_133 = None
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
        hidden_states_134 = getitem_98.expand(1, 8, 5, 19, 128)
        getitem_98 = None
        value_26 = hidden_states_134.reshape(1, 40, 19, 128)
        hidden_states_134 = None
        attention_mask_14 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_41 = attn_output_53.reshape(1, 19, -1)
        attn_output_53 = None
        attn_output_54 = reshape_41.contiguous()
        reshape_41 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_59 = hidden_states_137.to(torch.float16)
        hidden_states_137 = None
        hidden_states_138 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_59
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_59
        ) = None
        linear_95 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_95, inplace=False)
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_138,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_138 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_127 = silu_13 * linear_96
        silu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_127,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_127 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_61 = hidden_states_141.to(torch.float16)
        hidden_states_141 = None
        hidden_states_142 = (
            l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
            * to_61
        )
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            to_61
        ) = None
        linear_98 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_42 = linear_98.view((1, 19, -1, 128))
        linear_98 = None
        query_states_14 = view_42.transpose(1, 2)
        view_42 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_43 = linear_99.view((1, 19, -1, 128))
        linear_99 = None
        key_states_14 = view_43.transpose(1, 2)
        view_43 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_142 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_44 = linear_100.view((1, 19, -1, 128))
        linear_100 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_130 = query_states_14 * cos_17
        x1_28 = query_states_14[(Ellipsis, slice(None, 64, None))]
        x2_28 = query_states_14[(Ellipsis, slice(64, None, None))]
        query_states_14 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_131 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_130 + mul_131
        mul_130 = mul_131 = None
        mul_132 = key_states_14 * cos_17
        cos_17 = None
        x1_29 = key_states_14[(Ellipsis, slice(None, 64, None))]
        x2_29 = key_states_14[(Ellipsis, slice(64, None, None))]
        key_states_14 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_133 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_132 + mul_133
        mul_132 = mul_133 = None
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
        hidden_states_143 = getitem_104.expand(1, 8, 5, 19, 128)
        getitem_104 = None
        key_28 = hidden_states_143.reshape(1, 40, 19, 128)
        hidden_states_143 = None
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
        hidden_states_144 = getitem_105.expand(1, 8, 5, 19, 128)
        getitem_105 = None
        value_28 = hidden_states_144.reshape(1, 40, 19, 128)
        hidden_states_144 = None
        attention_mask_15 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_44 = attn_output_57.reshape(1, 19, -1)
        attn_output_57 = None
        attn_output_58 = reshape_44.contiguous()
        reshape_44 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_63 = hidden_states_147.to(torch.float16)
        hidden_states_147 = None
        hidden_states_148 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_63
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_63
        ) = None
        linear_102 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_102, inplace=False)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_148,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_148 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_136 = silu_14 * linear_103
        silu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_136,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_136 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_65 = hidden_states_151.to(torch.float16)
        hidden_states_151 = None
        hidden_states_152 = (
            l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
            * to_65
        )
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            to_65
        ) = None
        linear_105 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_45 = linear_105.view((1, 19, -1, 128))
        linear_105 = None
        query_states_15 = view_45.transpose(1, 2)
        view_45 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_46 = linear_106.view((1, 19, -1, 128))
        linear_106 = None
        key_states_15 = view_46.transpose(1, 2)
        view_46 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_152 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_47 = linear_107.view((1, 19, -1, 128))
        linear_107 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_139 = query_states_15 * cos_18
        x1_30 = query_states_15[(Ellipsis, slice(None, 64, None))]
        x2_30 = query_states_15[(Ellipsis, slice(64, None, None))]
        query_states_15 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_140 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_139 + mul_140
        mul_139 = mul_140 = None
        mul_141 = key_states_15 * cos_18
        cos_18 = None
        x1_31 = key_states_15[(Ellipsis, slice(None, 64, None))]
        x2_31 = key_states_15[(Ellipsis, slice(64, None, None))]
        key_states_15 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_142 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_141 + mul_142
        mul_141 = mul_142 = None
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
        hidden_states_153 = getitem_111.expand(1, 8, 5, 19, 128)
        getitem_111 = None
        key_30 = hidden_states_153.reshape(1, 40, 19, 128)
        hidden_states_153 = None
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
        hidden_states_154 = getitem_112.expand(1, 8, 5, 19, 128)
        getitem_112 = None
        value_30 = hidden_states_154.reshape(1, 40, 19, 128)
        hidden_states_154 = None
        attention_mask_16 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_47 = attn_output_61.reshape(1, 19, -1)
        attn_output_61 = None
        attn_output_62 = reshape_47.contiguous()
        reshape_47 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_67 = hidden_states_157.to(torch.float16)
        hidden_states_157 = None
        hidden_states_158 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_67
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_67
        ) = None
        linear_109 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_109, inplace=False)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_158,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_158 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_145 = silu_15 * linear_110
        silu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_145,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_145 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_69 = hidden_states_161.to(torch.float16)
        hidden_states_161 = None
        hidden_states_162 = (
            l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
            * to_69
        )
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            to_69
        ) = None
        linear_112 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_48 = linear_112.view((1, 19, -1, 128))
        linear_112 = None
        query_states_16 = view_48.transpose(1, 2)
        view_48 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_49 = linear_113.view((1, 19, -1, 128))
        linear_113 = None
        key_states_16 = view_49.transpose(1, 2)
        view_49 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_162,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_162 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_50 = linear_114.view((1, 19, -1, 128))
        linear_114 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_148 = query_states_16 * cos_19
        x1_32 = query_states_16[(Ellipsis, slice(None, 64, None))]
        x2_32 = query_states_16[(Ellipsis, slice(64, None, None))]
        query_states_16 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_149 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_148 + mul_149
        mul_148 = mul_149 = None
        mul_150 = key_states_16 * cos_19
        cos_19 = None
        x1_33 = key_states_16[(Ellipsis, slice(None, 64, None))]
        x2_33 = key_states_16[(Ellipsis, slice(64, None, None))]
        key_states_16 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_151 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_150 + mul_151
        mul_150 = mul_151 = None
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
        hidden_states_163 = getitem_118.expand(1, 8, 5, 19, 128)
        getitem_118 = None
        key_32 = hidden_states_163.reshape(1, 40, 19, 128)
        hidden_states_163 = None
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
        hidden_states_164 = getitem_119.expand(1, 8, 5, 19, 128)
        getitem_119 = None
        value_32 = hidden_states_164.reshape(1, 40, 19, 128)
        hidden_states_164 = None
        attention_mask_17 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_50 = attn_output_65.reshape(1, 19, -1)
        attn_output_65 = None
        attn_output_66 = reshape_50.contiguous()
        reshape_50 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_71 = hidden_states_167.to(torch.float16)
        hidden_states_167 = None
        hidden_states_168 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_71
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_71
        ) = None
        linear_116 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_116, inplace=False)
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_168 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_154 = silu_16 * linear_117
        silu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_154,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_154 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
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
        to_73 = hidden_states_171.to(torch.float16)
        hidden_states_171 = None
        hidden_states_172 = (
            l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
            * to_73
        )
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            to_73
        ) = None
        linear_119 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_51 = linear_119.view((1, 19, -1, 128))
        linear_119 = None
        query_states_17 = view_51.transpose(1, 2)
        view_51 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_52 = linear_120.view((1, 19, -1, 128))
        linear_120 = None
        key_states_17 = view_52.transpose(1, 2)
        view_52 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_172,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_172 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_53 = linear_121.view((1, 19, -1, 128))
        linear_121 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_157 = query_states_17 * cos_20
        x1_34 = query_states_17[(Ellipsis, slice(None, 64, None))]
        x2_34 = query_states_17[(Ellipsis, slice(64, None, None))]
        query_states_17 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_158 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_157 + mul_158
        mul_157 = mul_158 = None
        mul_159 = key_states_17 * cos_20
        cos_20 = None
        x1_35 = key_states_17[(Ellipsis, slice(None, 64, None))]
        x2_35 = key_states_17[(Ellipsis, slice(64, None, None))]
        key_states_17 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_160 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_159 + mul_160
        mul_159 = mul_160 = None
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
        hidden_states_173 = getitem_125.expand(1, 8, 5, 19, 128)
        getitem_125 = None
        key_34 = hidden_states_173.reshape(1, 40, 19, 128)
        hidden_states_173 = None
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
        hidden_states_174 = getitem_126.expand(1, 8, 5, 19, 128)
        getitem_126 = None
        value_34 = hidden_states_174.reshape(1, 40, 19, 128)
        hidden_states_174 = None
        attention_mask_18 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_53 = attn_output_69.reshape(1, 19, -1)
        attn_output_69 = None
        attn_output_70 = reshape_53.contiguous()
        reshape_53 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
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
        to_75 = hidden_states_177.to(torch.float16)
        hidden_states_177 = None
        hidden_states_178 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_75
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_75
        ) = None
        linear_123 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_123, inplace=False)
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_178,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_178 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_163 = silu_17 * linear_124
        silu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_163,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_163 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_179 = hidden_states_175 + down_proj_17
        hidden_states_175 = down_proj_17 = None
        hidden_states_180 = hidden_states_179.to(torch.float32)
        pow_37 = hidden_states_180.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_108 = variance_36 + 1e-05
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_181 = hidden_states_180 * rsqrt_36
        hidden_states_180 = rsqrt_36 = None
        to_77 = hidden_states_181.to(torch.float16)
        hidden_states_181 = None
        hidden_states_182 = (
            l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
            * to_77
        )
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            to_77
        ) = None
        linear_126 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_54 = linear_126.view((1, 19, -1, 128))
        linear_126 = None
        query_states_18 = view_54.transpose(1, 2)
        view_54 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_55 = linear_127.view((1, 19, -1, 128))
        linear_127 = None
        key_states_18 = view_55.transpose(1, 2)
        view_55 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_182,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_182 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_56 = linear_128.view((1, 19, -1, 128))
        linear_128 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_166 = query_states_18 * cos_21
        x1_36 = query_states_18[(Ellipsis, slice(None, 64, None))]
        x2_36 = query_states_18[(Ellipsis, slice(64, None, None))]
        query_states_18 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_37 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_167 = cat_37 * sin_21
        cat_37 = None
        q_embed_18 = mul_166 + mul_167
        mul_166 = mul_167 = None
        mul_168 = key_states_18 * cos_21
        cos_21 = None
        x1_37 = key_states_18[(Ellipsis, slice(None, 64, None))]
        x2_37 = key_states_18[(Ellipsis, slice(64, None, None))]
        key_states_18 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_38 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_169 = cat_38 * sin_21
        cat_38 = sin_21 = None
        k_embed_18 = mul_168 + mul_169
        mul_168 = mul_169 = None
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
        hidden_states_183 = getitem_132.expand(1, 8, 5, 19, 128)
        getitem_132 = None
        key_36 = hidden_states_183.reshape(1, 40, 19, 128)
        hidden_states_183 = None
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
        hidden_states_184 = getitem_133.expand(1, 8, 5, 19, 128)
        getitem_133 = None
        value_36 = hidden_states_184.reshape(1, 40, 19, 128)
        hidden_states_184 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_56 = attn_output_73.reshape(1, 19, -1)
        attn_output_73 = None
        attn_output_74 = reshape_56.contiguous()
        reshape_56 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_185 = hidden_states_179 + attn_output_75
        hidden_states_179 = attn_output_75 = None
        hidden_states_186 = hidden_states_185.to(torch.float32)
        pow_38 = hidden_states_186.pow(2)
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_112 = variance_37 + 1e-05
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_187 = hidden_states_186 * rsqrt_37
        hidden_states_186 = rsqrt_37 = None
        to_79 = hidden_states_187.to(torch.float16)
        hidden_states_187 = None
        hidden_states_188 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
            * to_79
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            to_79
        ) = None
        linear_130 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_130, inplace=False)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_188,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_188 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_172 = silu_18 * linear_131
        silu_18 = linear_131 = None
        down_proj_18 = torch._C._nn.linear(
            mul_172,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_172 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_189 = hidden_states_185 + down_proj_18
        hidden_states_185 = down_proj_18 = None
        hidden_states_190 = hidden_states_189.to(torch.float32)
        pow_39 = hidden_states_190.pow(2)
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_114 = variance_38 + 1e-05
        variance_38 = None
        rsqrt_38 = torch.rsqrt(add_114)
        add_114 = None
        hidden_states_191 = hidden_states_190 * rsqrt_38
        hidden_states_190 = rsqrt_38 = None
        to_81 = hidden_states_191.to(torch.float16)
        hidden_states_191 = None
        hidden_states_192 = (
            l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
            * to_81
        )
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            to_81
        ) = None
        linear_133 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_57 = linear_133.view((1, 19, -1, 128))
        linear_133 = None
        query_states_19 = view_57.transpose(1, 2)
        view_57 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_58 = linear_134.view((1, 19, -1, 128))
        linear_134 = None
        key_states_19 = view_58.transpose(1, 2)
        view_58 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_192 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_59 = linear_135.view((1, 19, -1, 128))
        linear_135 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_175 = query_states_19 * cos_22
        x1_38 = query_states_19[(Ellipsis, slice(None, 64, None))]
        x2_38 = query_states_19[(Ellipsis, slice(64, None, None))]
        query_states_19 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_39 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_176 = cat_39 * sin_22
        cat_39 = None
        q_embed_19 = mul_175 + mul_176
        mul_175 = mul_176 = None
        mul_177 = key_states_19 * cos_22
        cos_22 = None
        x1_39 = key_states_19[(Ellipsis, slice(None, 64, None))]
        x2_39 = key_states_19[(Ellipsis, slice(64, None, None))]
        key_states_19 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_40 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_178 = cat_40 * sin_22
        cat_40 = sin_22 = None
        k_embed_19 = mul_177 + mul_178
        mul_177 = mul_178 = None
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
        hidden_states_193 = getitem_139.expand(1, 8, 5, 19, 128)
        getitem_139 = None
        key_38 = hidden_states_193.reshape(1, 40, 19, 128)
        hidden_states_193 = None
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
        hidden_states_194 = getitem_140.expand(1, 8, 5, 19, 128)
        getitem_140 = None
        value_38 = hidden_states_194.reshape(1, 40, 19, 128)
        hidden_states_194 = None
        attention_mask_20 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_59 = attn_output_77.reshape(1, 19, -1)
        attn_output_77 = None
        attn_output_78 = reshape_59.contiguous()
        reshape_59 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_195 = hidden_states_189 + attn_output_79
        hidden_states_189 = attn_output_79 = None
        hidden_states_196 = hidden_states_195.to(torch.float32)
        pow_40 = hidden_states_196.pow(2)
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_118 = variance_39 + 1e-05
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_197 = hidden_states_196 * rsqrt_39
        hidden_states_196 = rsqrt_39 = None
        to_83 = hidden_states_197.to(torch.float16)
        hidden_states_197 = None
        hidden_states_198 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
            * to_83
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            to_83
        ) = None
        linear_137 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_137, inplace=False)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_198,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_198 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_181 = silu_19 * linear_138
        silu_19 = linear_138 = None
        down_proj_19 = torch._C._nn.linear(
            mul_181,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_181 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_199 = hidden_states_195 + down_proj_19
        hidden_states_195 = down_proj_19 = None
        hidden_states_200 = hidden_states_199.to(torch.float32)
        pow_41 = hidden_states_200.pow(2)
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_120 = variance_40 + 1e-05
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_201 = hidden_states_200 * rsqrt_40
        hidden_states_200 = rsqrt_40 = None
        to_85 = hidden_states_201.to(torch.float16)
        hidden_states_201 = None
        hidden_states_202 = (
            l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
            * to_85
        )
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            to_85
        ) = None
        linear_140 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_60 = linear_140.view((1, 19, -1, 128))
        linear_140 = None
        query_states_20 = view_60.transpose(1, 2)
        view_60 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_61 = linear_141.view((1, 19, -1, 128))
        linear_141 = None
        key_states_20 = view_61.transpose(1, 2)
        view_61 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_202,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_202 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_62 = linear_142.view((1, 19, -1, 128))
        linear_142 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_184 = query_states_20 * cos_23
        x1_40 = query_states_20[(Ellipsis, slice(None, 64, None))]
        x2_40 = query_states_20[(Ellipsis, slice(64, None, None))]
        query_states_20 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_41 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_185 = cat_41 * sin_23
        cat_41 = None
        q_embed_20 = mul_184 + mul_185
        mul_184 = mul_185 = None
        mul_186 = key_states_20 * cos_23
        cos_23 = None
        x1_41 = key_states_20[(Ellipsis, slice(None, 64, None))]
        x2_41 = key_states_20[(Ellipsis, slice(64, None, None))]
        key_states_20 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_42 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_187 = cat_42 * sin_23
        cat_42 = sin_23 = None
        k_embed_20 = mul_186 + mul_187
        mul_186 = mul_187 = None
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
        hidden_states_203 = getitem_146.expand(1, 8, 5, 19, 128)
        getitem_146 = None
        key_40 = hidden_states_203.reshape(1, 40, 19, 128)
        hidden_states_203 = None
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
        hidden_states_204 = getitem_147.expand(1, 8, 5, 19, 128)
        getitem_147 = None
        value_40 = hidden_states_204.reshape(1, 40, 19, 128)
        hidden_states_204 = None
        attention_mask_21 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_62 = attn_output_81.reshape(1, 19, -1)
        attn_output_81 = None
        attn_output_82 = reshape_62.contiguous()
        reshape_62 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_205 = hidden_states_199 + attn_output_83
        hidden_states_199 = attn_output_83 = None
        hidden_states_206 = hidden_states_205.to(torch.float32)
        pow_42 = hidden_states_206.pow(2)
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_124 = variance_41 + 1e-05
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_207 = hidden_states_206 * rsqrt_41
        hidden_states_206 = rsqrt_41 = None
        to_87 = hidden_states_207.to(torch.float16)
        hidden_states_207 = None
        hidden_states_208 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
            * to_87
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            to_87
        ) = None
        linear_144 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_144, inplace=False)
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_208,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_208 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_190 = silu_20 * linear_145
        silu_20 = linear_145 = None
        down_proj_20 = torch._C._nn.linear(
            mul_190,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_190 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_209 = hidden_states_205 + down_proj_20
        hidden_states_205 = down_proj_20 = None
        hidden_states_210 = hidden_states_209.to(torch.float32)
        pow_43 = hidden_states_210.pow(2)
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_126 = variance_42 + 1e-05
        variance_42 = None
        rsqrt_42 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_211 = hidden_states_210 * rsqrt_42
        hidden_states_210 = rsqrt_42 = None
        to_89 = hidden_states_211.to(torch.float16)
        hidden_states_211 = None
        hidden_states_212 = (
            l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
            * to_89
        )
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            to_89
        ) = None
        linear_147 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_63 = linear_147.view((1, 19, -1, 128))
        linear_147 = None
        query_states_21 = view_63.transpose(1, 2)
        view_63 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_64 = linear_148.view((1, 19, -1, 128))
        linear_148 = None
        key_states_21 = view_64.transpose(1, 2)
        view_64 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_212,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_212 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_65 = linear_149.view((1, 19, -1, 128))
        linear_149 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_193 = query_states_21 * cos_24
        x1_42 = query_states_21[(Ellipsis, slice(None, 64, None))]
        x2_42 = query_states_21[(Ellipsis, slice(64, None, None))]
        query_states_21 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_43 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_194 = cat_43 * sin_24
        cat_43 = None
        q_embed_21 = mul_193 + mul_194
        mul_193 = mul_194 = None
        mul_195 = key_states_21 * cos_24
        cos_24 = None
        x1_43 = key_states_21[(Ellipsis, slice(None, 64, None))]
        x2_43 = key_states_21[(Ellipsis, slice(64, None, None))]
        key_states_21 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_44 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_196 = cat_44 * sin_24
        cat_44 = sin_24 = None
        k_embed_21 = mul_195 + mul_196
        mul_195 = mul_196 = None
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
        hidden_states_213 = getitem_153.expand(1, 8, 5, 19, 128)
        getitem_153 = None
        key_42 = hidden_states_213.reshape(1, 40, 19, 128)
        hidden_states_213 = None
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
        hidden_states_214 = getitem_154.expand(1, 8, 5, 19, 128)
        getitem_154 = None
        value_42 = hidden_states_214.reshape(1, 40, 19, 128)
        hidden_states_214 = None
        attention_mask_22 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_65 = attn_output_85.reshape(1, 19, -1)
        attn_output_85 = None
        attn_output_86 = reshape_65.contiguous()
        reshape_65 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_215 = hidden_states_209 + attn_output_87
        hidden_states_209 = attn_output_87 = None
        hidden_states_216 = hidden_states_215.to(torch.float32)
        pow_44 = hidden_states_216.pow(2)
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_130 = variance_43 + 1e-05
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_130)
        add_130 = None
        hidden_states_217 = hidden_states_216 * rsqrt_43
        hidden_states_216 = rsqrt_43 = None
        to_91 = hidden_states_217.to(torch.float16)
        hidden_states_217 = None
        hidden_states_218 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
            * to_91
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            to_91
        ) = None
        linear_151 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_151, inplace=False)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_218,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_218 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_199 = silu_21 * linear_152
        silu_21 = linear_152 = None
        down_proj_21 = torch._C._nn.linear(
            mul_199,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_199 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_219 = hidden_states_215 + down_proj_21
        hidden_states_215 = down_proj_21 = None
        hidden_states_220 = hidden_states_219.to(torch.float32)
        pow_45 = hidden_states_220.pow(2)
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_132 = variance_44 + 1e-05
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_132)
        add_132 = None
        hidden_states_221 = hidden_states_220 * rsqrt_44
        hidden_states_220 = rsqrt_44 = None
        to_93 = hidden_states_221.to(torch.float16)
        hidden_states_221 = None
        hidden_states_222 = (
            l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
            * to_93
        )
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            to_93
        ) = None
        linear_154 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_66 = linear_154.view((1, 19, -1, 128))
        linear_154 = None
        query_states_22 = view_66.transpose(1, 2)
        view_66 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_67 = linear_155.view((1, 19, -1, 128))
        linear_155 = None
        key_states_22 = view_67.transpose(1, 2)
        view_67 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_222,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_222 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_68 = linear_156.view((1, 19, -1, 128))
        linear_156 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_202 = query_states_22 * cos_25
        x1_44 = query_states_22[(Ellipsis, slice(None, 64, None))]
        x2_44 = query_states_22[(Ellipsis, slice(64, None, None))]
        query_states_22 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_45 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_203 = cat_45 * sin_25
        cat_45 = None
        q_embed_22 = mul_202 + mul_203
        mul_202 = mul_203 = None
        mul_204 = key_states_22 * cos_25
        cos_25 = None
        x1_45 = key_states_22[(Ellipsis, slice(None, 64, None))]
        x2_45 = key_states_22[(Ellipsis, slice(64, None, None))]
        key_states_22 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_46 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_205 = cat_46 * sin_25
        cat_46 = sin_25 = None
        k_embed_22 = mul_204 + mul_205
        mul_204 = mul_205 = None
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
        hidden_states_223 = getitem_160.expand(1, 8, 5, 19, 128)
        getitem_160 = None
        key_44 = hidden_states_223.reshape(1, 40, 19, 128)
        hidden_states_223 = None
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
        hidden_states_224 = getitem_161.expand(1, 8, 5, 19, 128)
        getitem_161 = None
        value_44 = hidden_states_224.reshape(1, 40, 19, 128)
        hidden_states_224 = None
        attention_mask_23 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_68 = attn_output_89.reshape(1, 19, -1)
        attn_output_89 = None
        attn_output_90 = reshape_68.contiguous()
        reshape_68 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_225 = hidden_states_219 + attn_output_91
        hidden_states_219 = attn_output_91 = None
        hidden_states_226 = hidden_states_225.to(torch.float32)
        pow_46 = hidden_states_226.pow(2)
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_136 = variance_45 + 1e-05
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_227 = hidden_states_226 * rsqrt_45
        hidden_states_226 = rsqrt_45 = None
        to_95 = hidden_states_227.to(torch.float16)
        hidden_states_227 = None
        hidden_states_228 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
            * to_95
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            to_95
        ) = None
        linear_158 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_158, inplace=False)
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_228,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_228 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_208 = silu_22 * linear_159
        silu_22 = linear_159 = None
        down_proj_22 = torch._C._nn.linear(
            mul_208,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_208 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_229 = hidden_states_225 + down_proj_22
        hidden_states_225 = down_proj_22 = None
        hidden_states_230 = hidden_states_229.to(torch.float32)
        pow_47 = hidden_states_230.pow(2)
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_138 = variance_46 + 1e-05
        variance_46 = None
        rsqrt_46 = torch.rsqrt(add_138)
        add_138 = None
        hidden_states_231 = hidden_states_230 * rsqrt_46
        hidden_states_230 = rsqrt_46 = None
        to_97 = hidden_states_231.to(torch.float16)
        hidden_states_231 = None
        hidden_states_232 = (
            l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
            * to_97
        )
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            to_97
        ) = None
        linear_161 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_69 = linear_161.view((1, 19, -1, 128))
        linear_161 = None
        query_states_23 = view_69.transpose(1, 2)
        view_69 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_70 = linear_162.view((1, 19, -1, 128))
        linear_162 = None
        key_states_23 = view_70.transpose(1, 2)
        view_70 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_232,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_232 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_71 = linear_163.view((1, 19, -1, 128))
        linear_163 = None
        value_states_23 = view_71.transpose(1, 2)
        view_71 = None
        cos_26 = cos_2.unsqueeze(1)
        sin_26 = sin_2.unsqueeze(1)
        mul_211 = query_states_23 * cos_26
        x1_46 = query_states_23[(Ellipsis, slice(None, 64, None))]
        x2_46 = query_states_23[(Ellipsis, slice(64, None, None))]
        query_states_23 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_47 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_212 = cat_47 * sin_26
        cat_47 = None
        q_embed_23 = mul_211 + mul_212
        mul_211 = mul_212 = None
        mul_213 = key_states_23 * cos_26
        cos_26 = None
        x1_47 = key_states_23[(Ellipsis, slice(None, 64, None))]
        x2_47 = key_states_23[(Ellipsis, slice(64, None, None))]
        key_states_23 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_48 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_214 = cat_48 * sin_26
        cat_48 = sin_26 = None
        k_embed_23 = mul_213 + mul_214
        mul_213 = mul_214 = None
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
        hidden_states_233 = getitem_167.expand(1, 8, 5, 19, 128)
        getitem_167 = None
        key_46 = hidden_states_233.reshape(1, 40, 19, 128)
        hidden_states_233 = None
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
        hidden_states_234 = getitem_168.expand(1, 8, 5, 19, 128)
        getitem_168 = None
        value_46 = hidden_states_234.reshape(1, 40, 19, 128)
        hidden_states_234 = None
        attention_mask_24 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_71 = attn_output_93.reshape(1, 19, -1)
        attn_output_93 = None
        attn_output_94 = reshape_71.contiguous()
        reshape_71 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_235 = hidden_states_229 + attn_output_95
        hidden_states_229 = attn_output_95 = None
        hidden_states_236 = hidden_states_235.to(torch.float32)
        pow_48 = hidden_states_236.pow(2)
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_142 = variance_47 + 1e-05
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_237 = hidden_states_236 * rsqrt_47
        hidden_states_236 = rsqrt_47 = None
        to_99 = hidden_states_237.to(torch.float16)
        hidden_states_237 = None
        hidden_states_238 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
            * to_99
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            to_99
        ) = None
        linear_165 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_165, inplace=False)
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_238,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_238 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_217 = silu_23 * linear_166
        silu_23 = linear_166 = None
        down_proj_23 = torch._C._nn.linear(
            mul_217,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_217 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_239 = hidden_states_235 + down_proj_23
        hidden_states_235 = down_proj_23 = None
        hidden_states_240 = hidden_states_239.to(torch.float32)
        pow_49 = hidden_states_240.pow(2)
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_144 = variance_48 + 1e-05
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_241 = hidden_states_240 * rsqrt_48
        hidden_states_240 = rsqrt_48 = None
        to_101 = hidden_states_241.to(torch.float16)
        hidden_states_241 = None
        hidden_states_242 = (
            l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
            * to_101
        )
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            to_101
        ) = None
        linear_168 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_72 = linear_168.view((1, 19, -1, 128))
        linear_168 = None
        query_states_24 = view_72.transpose(1, 2)
        view_72 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_73 = linear_169.view((1, 19, -1, 128))
        linear_169 = None
        key_states_24 = view_73.transpose(1, 2)
        view_73 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_242,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_242 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_74 = linear_170.view((1, 19, -1, 128))
        linear_170 = None
        value_states_24 = view_74.transpose(1, 2)
        view_74 = None
        cos_27 = cos_2.unsqueeze(1)
        sin_27 = sin_2.unsqueeze(1)
        mul_220 = query_states_24 * cos_27
        x1_48 = query_states_24[(Ellipsis, slice(None, 64, None))]
        x2_48 = query_states_24[(Ellipsis, slice(64, None, None))]
        query_states_24 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_49 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_221 = cat_49 * sin_27
        cat_49 = None
        q_embed_24 = mul_220 + mul_221
        mul_220 = mul_221 = None
        mul_222 = key_states_24 * cos_27
        cos_27 = None
        x1_49 = key_states_24[(Ellipsis, slice(None, 64, None))]
        x2_49 = key_states_24[(Ellipsis, slice(64, None, None))]
        key_states_24 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_50 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_223 = cat_50 * sin_27
        cat_50 = sin_27 = None
        k_embed_24 = mul_222 + mul_223
        mul_222 = mul_223 = None
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
        hidden_states_243 = getitem_174.expand(1, 8, 5, 19, 128)
        getitem_174 = None
        key_48 = hidden_states_243.reshape(1, 40, 19, 128)
        hidden_states_243 = None
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
        hidden_states_244 = getitem_175.expand(1, 8, 5, 19, 128)
        getitem_175 = None
        value_48 = hidden_states_244.reshape(1, 40, 19, 128)
        hidden_states_244 = None
        attention_mask_25 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_74 = attn_output_97.reshape(1, 19, -1)
        attn_output_97 = None
        attn_output_98 = reshape_74.contiguous()
        reshape_74 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_245 = hidden_states_239 + attn_output_99
        hidden_states_239 = attn_output_99 = None
        hidden_states_246 = hidden_states_245.to(torch.float32)
        pow_50 = hidden_states_246.pow(2)
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_148 = variance_49 + 1e-05
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_148)
        add_148 = None
        hidden_states_247 = hidden_states_246 * rsqrt_49
        hidden_states_246 = rsqrt_49 = None
        to_103 = hidden_states_247.to(torch.float16)
        hidden_states_247 = None
        hidden_states_248 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
            * to_103
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            to_103
        ) = None
        linear_172 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_172, inplace=False)
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_248,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_248 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_226 = silu_24 * linear_173
        silu_24 = linear_173 = None
        down_proj_24 = torch._C._nn.linear(
            mul_226,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_226 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_249 = hidden_states_245 + down_proj_24
        hidden_states_245 = down_proj_24 = None
        hidden_states_250 = hidden_states_249.to(torch.float32)
        pow_51 = hidden_states_250.pow(2)
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_150 = variance_50 + 1e-05
        variance_50 = None
        rsqrt_50 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_251 = hidden_states_250 * rsqrt_50
        hidden_states_250 = rsqrt_50 = None
        to_105 = hidden_states_251.to(torch.float16)
        hidden_states_251 = None
        hidden_states_252 = (
            l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
            * to_105
        )
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            to_105
        ) = None
        linear_175 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_75 = linear_175.view((1, 19, -1, 128))
        linear_175 = None
        query_states_25 = view_75.transpose(1, 2)
        view_75 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_76 = linear_176.view((1, 19, -1, 128))
        linear_176 = None
        key_states_25 = view_76.transpose(1, 2)
        view_76 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_252,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_252 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_77 = linear_177.view((1, 19, -1, 128))
        linear_177 = None
        value_states_25 = view_77.transpose(1, 2)
        view_77 = None
        cos_28 = cos_2.unsqueeze(1)
        sin_28 = sin_2.unsqueeze(1)
        mul_229 = query_states_25 * cos_28
        x1_50 = query_states_25[(Ellipsis, slice(None, 64, None))]
        x2_50 = query_states_25[(Ellipsis, slice(64, None, None))]
        query_states_25 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_51 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_230 = cat_51 * sin_28
        cat_51 = None
        q_embed_25 = mul_229 + mul_230
        mul_229 = mul_230 = None
        mul_231 = key_states_25 * cos_28
        cos_28 = None
        x1_51 = key_states_25[(Ellipsis, slice(None, 64, None))]
        x2_51 = key_states_25[(Ellipsis, slice(64, None, None))]
        key_states_25 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_52 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_232 = cat_52 * sin_28
        cat_52 = sin_28 = None
        k_embed_25 = mul_231 + mul_232
        mul_231 = mul_232 = None
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
        hidden_states_253 = getitem_181.expand(1, 8, 5, 19, 128)
        getitem_181 = None
        key_50 = hidden_states_253.reshape(1, 40, 19, 128)
        hidden_states_253 = None
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
        hidden_states_254 = getitem_182.expand(1, 8, 5, 19, 128)
        getitem_182 = None
        value_50 = hidden_states_254.reshape(1, 40, 19, 128)
        hidden_states_254 = None
        attention_mask_26 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_77 = attn_output_101.reshape(1, 19, -1)
        attn_output_101 = None
        attn_output_102 = reshape_77.contiguous()
        reshape_77 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_255 = hidden_states_249 + attn_output_103
        hidden_states_249 = attn_output_103 = None
        hidden_states_256 = hidden_states_255.to(torch.float32)
        pow_52 = hidden_states_256.pow(2)
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_154 = variance_51 + 1e-05
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_154)
        add_154 = None
        hidden_states_257 = hidden_states_256 * rsqrt_51
        hidden_states_256 = rsqrt_51 = None
        to_107 = hidden_states_257.to(torch.float16)
        hidden_states_257 = None
        hidden_states_258 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
            * to_107
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            to_107
        ) = None
        linear_179 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_179, inplace=False)
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_258,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_258 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_235 = silu_25 * linear_180
        silu_25 = linear_180 = None
        down_proj_25 = torch._C._nn.linear(
            mul_235,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_235 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_259 = hidden_states_255 + down_proj_25
        hidden_states_255 = down_proj_25 = None
        hidden_states_260 = hidden_states_259.to(torch.float32)
        pow_53 = hidden_states_260.pow(2)
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_156 = variance_52 + 1e-05
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_156)
        add_156 = None
        hidden_states_261 = hidden_states_260 * rsqrt_52
        hidden_states_260 = rsqrt_52 = None
        to_109 = hidden_states_261.to(torch.float16)
        hidden_states_261 = None
        hidden_states_262 = (
            l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
            * to_109
        )
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = (
            to_109
        ) = None
        linear_182 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_78 = linear_182.view((1, 19, -1, 128))
        linear_182 = None
        query_states_26 = view_78.transpose(1, 2)
        view_78 = None
        linear_183 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_79 = linear_183.view((1, 19, -1, 128))
        linear_183 = None
        key_states_26 = view_79.transpose(1, 2)
        view_79 = None
        linear_184 = torch._C._nn.linear(
            hidden_states_262,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_262 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_80 = linear_184.view((1, 19, -1, 128))
        linear_184 = None
        value_states_26 = view_80.transpose(1, 2)
        view_80 = None
        cos_29 = cos_2.unsqueeze(1)
        sin_29 = sin_2.unsqueeze(1)
        mul_238 = query_states_26 * cos_29
        x1_52 = query_states_26[(Ellipsis, slice(None, 64, None))]
        x2_52 = query_states_26[(Ellipsis, slice(64, None, None))]
        query_states_26 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_53 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_239 = cat_53 * sin_29
        cat_53 = None
        q_embed_26 = mul_238 + mul_239
        mul_238 = mul_239 = None
        mul_240 = key_states_26 * cos_29
        cos_29 = None
        x1_53 = key_states_26[(Ellipsis, slice(None, 64, None))]
        x2_53 = key_states_26[(Ellipsis, slice(64, None, None))]
        key_states_26 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_54 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_241 = cat_54 * sin_29
        cat_54 = sin_29 = None
        k_embed_26 = mul_240 + mul_241
        mul_240 = mul_241 = None
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
        hidden_states_263 = getitem_188.expand(1, 8, 5, 19, 128)
        getitem_188 = None
        key_52 = hidden_states_263.reshape(1, 40, 19, 128)
        hidden_states_263 = None
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
        hidden_states_264 = getitem_189.expand(1, 8, 5, 19, 128)
        getitem_189 = None
        value_52 = hidden_states_264.reshape(1, 40, 19, 128)
        hidden_states_264 = None
        attention_mask_27 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_80 = attn_output_105.reshape(1, 19, -1)
        attn_output_105 = None
        attn_output_106 = reshape_80.contiguous()
        reshape_80 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_265 = hidden_states_259 + attn_output_107
        hidden_states_259 = attn_output_107 = None
        hidden_states_266 = hidden_states_265.to(torch.float32)
        pow_54 = hidden_states_266.pow(2)
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_160 = variance_53 + 1e-05
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_267 = hidden_states_266 * rsqrt_53
        hidden_states_266 = rsqrt_53 = None
        to_111 = hidden_states_267.to(torch.float16)
        hidden_states_267 = None
        hidden_states_268 = (
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
            * to_111
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = (
            to_111
        ) = None
        linear_186 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_186, inplace=False)
        linear_186 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_268,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_268 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_244 = silu_26 * linear_187
        silu_26 = linear_187 = None
        down_proj_26 = torch._C._nn.linear(
            mul_244,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_244 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_269 = hidden_states_265 + down_proj_26
        hidden_states_265 = down_proj_26 = None
        hidden_states_270 = hidden_states_269.to(torch.float32)
        pow_55 = hidden_states_270.pow(2)
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_162 = variance_54 + 1e-05
        variance_54 = None
        rsqrt_54 = torch.rsqrt(add_162)
        add_162 = None
        hidden_states_271 = hidden_states_270 * rsqrt_54
        hidden_states_270 = rsqrt_54 = None
        to_113 = hidden_states_271.to(torch.float16)
        hidden_states_271 = None
        hidden_states_272 = (
            l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
            * to_113
        )
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = (
            to_113
        ) = None
        linear_189 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_81 = linear_189.view((1, 19, -1, 128))
        linear_189 = None
        query_states_27 = view_81.transpose(1, 2)
        view_81 = None
        linear_190 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_82 = linear_190.view((1, 19, -1, 128))
        linear_190 = None
        key_states_27 = view_82.transpose(1, 2)
        view_82 = None
        linear_191 = torch._C._nn.linear(
            hidden_states_272,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_272 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_83 = linear_191.view((1, 19, -1, 128))
        linear_191 = None
        value_states_27 = view_83.transpose(1, 2)
        view_83 = None
        cos_30 = cos_2.unsqueeze(1)
        sin_30 = sin_2.unsqueeze(1)
        mul_247 = query_states_27 * cos_30
        x1_54 = query_states_27[(Ellipsis, slice(None, 64, None))]
        x2_54 = query_states_27[(Ellipsis, slice(64, None, None))]
        query_states_27 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_55 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_248 = cat_55 * sin_30
        cat_55 = None
        q_embed_27 = mul_247 + mul_248
        mul_247 = mul_248 = None
        mul_249 = key_states_27 * cos_30
        cos_30 = None
        x1_55 = key_states_27[(Ellipsis, slice(None, 64, None))]
        x2_55 = key_states_27[(Ellipsis, slice(64, None, None))]
        key_states_27 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_56 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_250 = cat_56 * sin_30
        cat_56 = sin_30 = None
        k_embed_27 = mul_249 + mul_250
        mul_249 = mul_250 = None
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
        hidden_states_273 = getitem_195.expand(1, 8, 5, 19, 128)
        getitem_195 = None
        key_54 = hidden_states_273.reshape(1, 40, 19, 128)
        hidden_states_273 = None
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
        hidden_states_274 = getitem_196.expand(1, 8, 5, 19, 128)
        getitem_196 = None
        value_54 = hidden_states_274.reshape(1, 40, 19, 128)
        hidden_states_274 = None
        attention_mask_28 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_83 = attn_output_109.reshape(1, 19, -1)
        attn_output_109 = None
        attn_output_110 = reshape_83.contiguous()
        reshape_83 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_275 = hidden_states_269 + attn_output_111
        hidden_states_269 = attn_output_111 = None
        hidden_states_276 = hidden_states_275.to(torch.float32)
        pow_56 = hidden_states_276.pow(2)
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_166 = variance_55 + 1e-05
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_277 = hidden_states_276 * rsqrt_55
        hidden_states_276 = rsqrt_55 = None
        to_115 = hidden_states_277.to(torch.float16)
        hidden_states_277 = None
        hidden_states_278 = (
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
            * to_115
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = (
            to_115
        ) = None
        linear_193 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_193, inplace=False)
        linear_193 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_278,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_278 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_253 = silu_27 * linear_194
        silu_27 = linear_194 = None
        down_proj_27 = torch._C._nn.linear(
            mul_253,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_253 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_279 = hidden_states_275 + down_proj_27
        hidden_states_275 = down_proj_27 = None
        hidden_states_280 = hidden_states_279.to(torch.float32)
        pow_57 = hidden_states_280.pow(2)
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_168 = variance_56 + 1e-05
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_281 = hidden_states_280 * rsqrt_56
        hidden_states_280 = rsqrt_56 = None
        to_117 = hidden_states_281.to(torch.float16)
        hidden_states_281 = None
        hidden_states_282 = (
            l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
            * to_117
        )
        l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = (
            to_117
        ) = None
        linear_196 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_84 = linear_196.view((1, 19, -1, 128))
        linear_196 = None
        query_states_28 = view_84.transpose(1, 2)
        view_84 = None
        linear_197 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_85 = linear_197.view((1, 19, -1, 128))
        linear_197 = None
        key_states_28 = view_85.transpose(1, 2)
        view_85 = None
        linear_198 = torch._C._nn.linear(
            hidden_states_282,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_282 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_86 = linear_198.view((1, 19, -1, 128))
        linear_198 = None
        value_states_28 = view_86.transpose(1, 2)
        view_86 = None
        cos_31 = cos_2.unsqueeze(1)
        sin_31 = sin_2.unsqueeze(1)
        mul_256 = query_states_28 * cos_31
        x1_56 = query_states_28[(Ellipsis, slice(None, 64, None))]
        x2_56 = query_states_28[(Ellipsis, slice(64, None, None))]
        query_states_28 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_57 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_257 = cat_57 * sin_31
        cat_57 = None
        q_embed_28 = mul_256 + mul_257
        mul_256 = mul_257 = None
        mul_258 = key_states_28 * cos_31
        cos_31 = None
        x1_57 = key_states_28[(Ellipsis, slice(None, 64, None))]
        x2_57 = key_states_28[(Ellipsis, slice(64, None, None))]
        key_states_28 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_58 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_259 = cat_58 * sin_31
        cat_58 = sin_31 = None
        k_embed_28 = mul_258 + mul_259
        mul_258 = mul_259 = None
        getitem_202 = k_embed_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_28 = None
        hidden_states_283 = getitem_202.expand(1, 8, 5, 19, 128)
        getitem_202 = None
        key_56 = hidden_states_283.reshape(1, 40, 19, 128)
        hidden_states_283 = None
        getitem_203 = value_states_28[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_28 = None
        hidden_states_284 = getitem_203.expand(1, 8, 5, 19, 128)
        getitem_203 = None
        value_56 = hidden_states_284.reshape(1, 40, 19, 128)
        hidden_states_284 = None
        attention_mask_29 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_86 = attn_output_113.reshape(1, 19, -1)
        attn_output_113 = None
        attn_output_114 = reshape_86.contiguous()
        reshape_86 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_285 = hidden_states_279 + attn_output_115
        hidden_states_279 = attn_output_115 = None
        hidden_states_286 = hidden_states_285.to(torch.float32)
        pow_58 = hidden_states_286.pow(2)
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_172 = variance_57 + 1e-05
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_172)
        add_172 = None
        hidden_states_287 = hidden_states_286 * rsqrt_57
        hidden_states_286 = rsqrt_57 = None
        to_119 = hidden_states_287.to(torch.float16)
        hidden_states_287 = None
        hidden_states_288 = (
            l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
            * to_119
        )
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = (
            to_119
        ) = None
        linear_200 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_28 = torch.nn.functional.silu(linear_200, inplace=False)
        linear_200 = None
        linear_201 = torch._C._nn.linear(
            hidden_states_288,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_288 = l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_262 = silu_28 * linear_201
        silu_28 = linear_201 = None
        down_proj_28 = torch._C._nn.linear(
            mul_262,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_262 = l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_289 = hidden_states_285 + down_proj_28
        hidden_states_285 = down_proj_28 = None
        hidden_states_290 = hidden_states_289.to(torch.float32)
        pow_59 = hidden_states_290.pow(2)
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_174 = variance_58 + 1e-05
        variance_58 = None
        rsqrt_58 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_291 = hidden_states_290 * rsqrt_58
        hidden_states_290 = rsqrt_58 = None
        to_121 = hidden_states_291.to(torch.float16)
        hidden_states_291 = None
        hidden_states_292 = (
            l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
            * to_121
        )
        l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = (
            to_121
        ) = None
        linear_203 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_87 = linear_203.view((1, 19, -1, 128))
        linear_203 = None
        query_states_29 = view_87.transpose(1, 2)
        view_87 = None
        linear_204 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_88 = linear_204.view((1, 19, -1, 128))
        linear_204 = None
        key_states_29 = view_88.transpose(1, 2)
        view_88 = None
        linear_205 = torch._C._nn.linear(
            hidden_states_292,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_292 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_89 = linear_205.view((1, 19, -1, 128))
        linear_205 = None
        value_states_29 = view_89.transpose(1, 2)
        view_89 = None
        cos_32 = cos_2.unsqueeze(1)
        sin_32 = sin_2.unsqueeze(1)
        mul_265 = query_states_29 * cos_32
        x1_58 = query_states_29[(Ellipsis, slice(None, 64, None))]
        x2_58 = query_states_29[(Ellipsis, slice(64, None, None))]
        query_states_29 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_59 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_266 = cat_59 * sin_32
        cat_59 = None
        q_embed_29 = mul_265 + mul_266
        mul_265 = mul_266 = None
        mul_267 = key_states_29 * cos_32
        cos_32 = None
        x1_59 = key_states_29[(Ellipsis, slice(None, 64, None))]
        x2_59 = key_states_29[(Ellipsis, slice(64, None, None))]
        key_states_29 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_60 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_268 = cat_60 * sin_32
        cat_60 = sin_32 = None
        k_embed_29 = mul_267 + mul_268
        mul_267 = mul_268 = None
        getitem_209 = k_embed_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_29 = None
        hidden_states_293 = getitem_209.expand(1, 8, 5, 19, 128)
        getitem_209 = None
        key_58 = hidden_states_293.reshape(1, 40, 19, 128)
        hidden_states_293 = None
        getitem_210 = value_states_29[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_29 = None
        hidden_states_294 = getitem_210.expand(1, 8, 5, 19, 128)
        getitem_210 = None
        value_58 = hidden_states_294.reshape(1, 40, 19, 128)
        hidden_states_294 = None
        attention_mask_30 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_89 = attn_output_117.reshape(1, 19, -1)
        attn_output_117 = None
        attn_output_118 = reshape_89.contiguous()
        reshape_89 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_295 = hidden_states_289 + attn_output_119
        hidden_states_289 = attn_output_119 = None
        hidden_states_296 = hidden_states_295.to(torch.float32)
        pow_60 = hidden_states_296.pow(2)
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_178 = variance_59 + 1e-05
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_178)
        add_178 = None
        hidden_states_297 = hidden_states_296 * rsqrt_59
        hidden_states_296 = rsqrt_59 = None
        to_123 = hidden_states_297.to(torch.float16)
        hidden_states_297 = None
        hidden_states_298 = (
            l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
            * to_123
        )
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = (
            to_123
        ) = None
        linear_207 = torch._C._nn.linear(
            hidden_states_298,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_29 = torch.nn.functional.silu(linear_207, inplace=False)
        linear_207 = None
        linear_208 = torch._C._nn.linear(
            hidden_states_298,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_298 = l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_271 = silu_29 * linear_208
        silu_29 = linear_208 = None
        down_proj_29 = torch._C._nn.linear(
            mul_271,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_271 = l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_299 = hidden_states_295 + down_proj_29
        hidden_states_295 = down_proj_29 = None
        hidden_states_300 = hidden_states_299.to(torch.float32)
        pow_61 = hidden_states_300.pow(2)
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_180 = variance_60 + 1e-05
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_180)
        add_180 = None
        hidden_states_301 = hidden_states_300 * rsqrt_60
        hidden_states_300 = rsqrt_60 = None
        to_125 = hidden_states_301.to(torch.float16)
        hidden_states_301 = None
        hidden_states_302 = (
            l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
            * to_125
        )
        l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = (
            to_125
        ) = None
        linear_210 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_90 = linear_210.view((1, 19, -1, 128))
        linear_210 = None
        query_states_30 = view_90.transpose(1, 2)
        view_90 = None
        linear_211 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_91 = linear_211.view((1, 19, -1, 128))
        linear_211 = None
        key_states_30 = view_91.transpose(1, 2)
        view_91 = None
        linear_212 = torch._C._nn.linear(
            hidden_states_302,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_302 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_92 = linear_212.view((1, 19, -1, 128))
        linear_212 = None
        value_states_30 = view_92.transpose(1, 2)
        view_92 = None
        cos_33 = cos_2.unsqueeze(1)
        sin_33 = sin_2.unsqueeze(1)
        mul_274 = query_states_30 * cos_33
        x1_60 = query_states_30[(Ellipsis, slice(None, 64, None))]
        x2_60 = query_states_30[(Ellipsis, slice(64, None, None))]
        query_states_30 = None
        neg_60 = -x2_60
        x2_60 = None
        cat_61 = torch.cat((neg_60, x1_60), dim=-1)
        neg_60 = x1_60 = None
        mul_275 = cat_61 * sin_33
        cat_61 = None
        q_embed_30 = mul_274 + mul_275
        mul_274 = mul_275 = None
        mul_276 = key_states_30 * cos_33
        cos_33 = None
        x1_61 = key_states_30[(Ellipsis, slice(None, 64, None))]
        x2_61 = key_states_30[(Ellipsis, slice(64, None, None))]
        key_states_30 = None
        neg_61 = -x2_61
        x2_61 = None
        cat_62 = torch.cat((neg_61, x1_61), dim=-1)
        neg_61 = x1_61 = None
        mul_277 = cat_62 * sin_33
        cat_62 = sin_33 = None
        k_embed_30 = mul_276 + mul_277
        mul_276 = mul_277 = None
        getitem_216 = k_embed_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_30 = None
        hidden_states_303 = getitem_216.expand(1, 8, 5, 19, 128)
        getitem_216 = None
        key_60 = hidden_states_303.reshape(1, 40, 19, 128)
        hidden_states_303 = None
        getitem_217 = value_states_30[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_30 = None
        hidden_states_304 = getitem_217.expand(1, 8, 5, 19, 128)
        getitem_217 = None
        value_60 = hidden_states_304.reshape(1, 40, 19, 128)
        hidden_states_304 = None
        attention_mask_31 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_92 = attn_output_121.reshape(1, 19, -1)
        attn_output_121 = None
        attn_output_122 = reshape_92.contiguous()
        reshape_92 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_305 = hidden_states_299 + attn_output_123
        hidden_states_299 = attn_output_123 = None
        hidden_states_306 = hidden_states_305.to(torch.float32)
        pow_62 = hidden_states_306.pow(2)
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_184 = variance_61 + 1e-05
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_307 = hidden_states_306 * rsqrt_61
        hidden_states_306 = rsqrt_61 = None
        to_127 = hidden_states_307.to(torch.float16)
        hidden_states_307 = None
        hidden_states_308 = (
            l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
            * to_127
        )
        l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = (
            to_127
        ) = None
        linear_214 = torch._C._nn.linear(
            hidden_states_308,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_30 = torch.nn.functional.silu(linear_214, inplace=False)
        linear_214 = None
        linear_215 = torch._C._nn.linear(
            hidden_states_308,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_308 = l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_280 = silu_30 * linear_215
        silu_30 = linear_215 = None
        down_proj_30 = torch._C._nn.linear(
            mul_280,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_280 = l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_309 = hidden_states_305 + down_proj_30
        hidden_states_305 = down_proj_30 = None
        hidden_states_310 = hidden_states_309.to(torch.float32)
        pow_63 = hidden_states_310.pow(2)
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_186 = variance_62 + 1e-05
        variance_62 = None
        rsqrt_62 = torch.rsqrt(add_186)
        add_186 = None
        hidden_states_311 = hidden_states_310 * rsqrt_62
        hidden_states_310 = rsqrt_62 = None
        to_129 = hidden_states_311.to(torch.float16)
        hidden_states_311 = None
        hidden_states_312 = (
            l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
            * to_129
        )
        l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = (
            to_129
        ) = None
        linear_217 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_93 = linear_217.view((1, 19, -1, 128))
        linear_217 = None
        query_states_31 = view_93.transpose(1, 2)
        view_93 = None
        linear_218 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_94 = linear_218.view((1, 19, -1, 128))
        linear_218 = None
        key_states_31 = view_94.transpose(1, 2)
        view_94 = None
        linear_219 = torch._C._nn.linear(
            hidden_states_312,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_312 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_95 = linear_219.view((1, 19, -1, 128))
        linear_219 = None
        value_states_31 = view_95.transpose(1, 2)
        view_95 = None
        cos_34 = cos_2.unsqueeze(1)
        sin_34 = sin_2.unsqueeze(1)
        mul_283 = query_states_31 * cos_34
        x1_62 = query_states_31[(Ellipsis, slice(None, 64, None))]
        x2_62 = query_states_31[(Ellipsis, slice(64, None, None))]
        query_states_31 = None
        neg_62 = -x2_62
        x2_62 = None
        cat_63 = torch.cat((neg_62, x1_62), dim=-1)
        neg_62 = x1_62 = None
        mul_284 = cat_63 * sin_34
        cat_63 = None
        q_embed_31 = mul_283 + mul_284
        mul_283 = mul_284 = None
        mul_285 = key_states_31 * cos_34
        cos_34 = None
        x1_63 = key_states_31[(Ellipsis, slice(None, 64, None))]
        x2_63 = key_states_31[(Ellipsis, slice(64, None, None))]
        key_states_31 = None
        neg_63 = -x2_63
        x2_63 = None
        cat_64 = torch.cat((neg_63, x1_63), dim=-1)
        neg_63 = x1_63 = None
        mul_286 = cat_64 * sin_34
        cat_64 = sin_34 = None
        k_embed_31 = mul_285 + mul_286
        mul_285 = mul_286 = None
        getitem_223 = k_embed_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_31 = None
        hidden_states_313 = getitem_223.expand(1, 8, 5, 19, 128)
        getitem_223 = None
        key_62 = hidden_states_313.reshape(1, 40, 19, 128)
        hidden_states_313 = None
        getitem_224 = value_states_31[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_31 = None
        hidden_states_314 = getitem_224.expand(1, 8, 5, 19, 128)
        getitem_224 = None
        value_62 = hidden_states_314.reshape(1, 40, 19, 128)
        hidden_states_314 = None
        attention_mask_32 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_95 = attn_output_125.reshape(1, 19, -1)
        attn_output_125 = None
        attn_output_126 = reshape_95.contiguous()
        reshape_95 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_315 = hidden_states_309 + attn_output_127
        hidden_states_309 = attn_output_127 = None
        hidden_states_316 = hidden_states_315.to(torch.float32)
        pow_64 = hidden_states_316.pow(2)
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_190 = variance_63 + 1e-05
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_317 = hidden_states_316 * rsqrt_63
        hidden_states_316 = rsqrt_63 = None
        to_131 = hidden_states_317.to(torch.float16)
        hidden_states_317 = None
        hidden_states_318 = (
            l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
            * to_131
        )
        l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = (
            to_131
        ) = None
        linear_221 = torch._C._nn.linear(
            hidden_states_318,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_31 = torch.nn.functional.silu(linear_221, inplace=False)
        linear_221 = None
        linear_222 = torch._C._nn.linear(
            hidden_states_318,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_318 = l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_289 = silu_31 * linear_222
        silu_31 = linear_222 = None
        down_proj_31 = torch._C._nn.linear(
            mul_289,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_289 = l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_319 = hidden_states_315 + down_proj_31
        hidden_states_315 = down_proj_31 = None
        hidden_states_320 = hidden_states_319.to(torch.float32)
        pow_65 = hidden_states_320.pow(2)
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_192 = variance_64 + 1e-05
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_321 = hidden_states_320 * rsqrt_64
        hidden_states_320 = rsqrt_64 = None
        to_133 = hidden_states_321.to(torch.float16)
        hidden_states_321 = None
        hidden_states_322 = (
            l_self_modules_model_modules_layers_modules_32_modules_input_layernorm_parameters_weight_
            * to_133
        )
        l_self_modules_model_modules_layers_modules_32_modules_input_layernorm_parameters_weight_ = (
            to_133
        ) = None
        linear_224 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_96 = linear_224.view((1, 19, -1, 128))
        linear_224 = None
        query_states_32 = view_96.transpose(1, 2)
        view_96 = None
        linear_225 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_97 = linear_225.view((1, 19, -1, 128))
        linear_225 = None
        key_states_32 = view_97.transpose(1, 2)
        view_97 = None
        linear_226 = torch._C._nn.linear(
            hidden_states_322,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_322 = l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_98 = linear_226.view((1, 19, -1, 128))
        linear_226 = None
        value_states_32 = view_98.transpose(1, 2)
        view_98 = None
        cos_35 = cos_2.unsqueeze(1)
        sin_35 = sin_2.unsqueeze(1)
        mul_292 = query_states_32 * cos_35
        x1_64 = query_states_32[(Ellipsis, slice(None, 64, None))]
        x2_64 = query_states_32[(Ellipsis, slice(64, None, None))]
        query_states_32 = None
        neg_64 = -x2_64
        x2_64 = None
        cat_65 = torch.cat((neg_64, x1_64), dim=-1)
        neg_64 = x1_64 = None
        mul_293 = cat_65 * sin_35
        cat_65 = None
        q_embed_32 = mul_292 + mul_293
        mul_292 = mul_293 = None
        mul_294 = key_states_32 * cos_35
        cos_35 = None
        x1_65 = key_states_32[(Ellipsis, slice(None, 64, None))]
        x2_65 = key_states_32[(Ellipsis, slice(64, None, None))]
        key_states_32 = None
        neg_65 = -x2_65
        x2_65 = None
        cat_66 = torch.cat((neg_65, x1_65), dim=-1)
        neg_65 = x1_65 = None
        mul_295 = cat_66 * sin_35
        cat_66 = sin_35 = None
        k_embed_32 = mul_294 + mul_295
        mul_294 = mul_295 = None
        getitem_230 = k_embed_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_32 = None
        hidden_states_323 = getitem_230.expand(1, 8, 5, 19, 128)
        getitem_230 = None
        key_64 = hidden_states_323.reshape(1, 40, 19, 128)
        hidden_states_323 = None
        getitem_231 = value_states_32[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_32 = None
        hidden_states_324 = getitem_231.expand(1, 8, 5, 19, 128)
        getitem_231 = None
        value_64 = hidden_states_324.reshape(1, 40, 19, 128)
        hidden_states_324 = None
        attention_mask_33 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_98 = attn_output_129.reshape(1, 19, -1)
        attn_output_129 = None
        attn_output_130 = reshape_98.contiguous()
        reshape_98 = None
        attn_output_131 = torch._C._nn.linear(
            attn_output_130,
            l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_130 = l_self_modules_model_modules_layers_modules_32_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_325 = hidden_states_319 + attn_output_131
        hidden_states_319 = attn_output_131 = None
        hidden_states_326 = hidden_states_325.to(torch.float32)
        pow_66 = hidden_states_326.pow(2)
        variance_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_196 = variance_65 + 1e-05
        variance_65 = None
        rsqrt_65 = torch.rsqrt(add_196)
        add_196 = None
        hidden_states_327 = hidden_states_326 * rsqrt_65
        hidden_states_326 = rsqrt_65 = None
        to_135 = hidden_states_327.to(torch.float16)
        hidden_states_327 = None
        hidden_states_328 = (
            l_self_modules_model_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_
            * to_135
        )
        l_self_modules_model_modules_layers_modules_32_modules_post_attention_layernorm_parameters_weight_ = (
            to_135
        ) = None
        linear_228 = torch._C._nn.linear(
            hidden_states_328,
            l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_32 = torch.nn.functional.silu(linear_228, inplace=False)
        linear_228 = None
        linear_229 = torch._C._nn.linear(
            hidden_states_328,
            l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_328 = l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_298 = silu_32 * linear_229
        silu_32 = linear_229 = None
        down_proj_32 = torch._C._nn.linear(
            mul_298,
            l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_298 = l_self_modules_model_modules_layers_modules_32_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_329 = hidden_states_325 + down_proj_32
        hidden_states_325 = down_proj_32 = None
        hidden_states_330 = hidden_states_329.to(torch.float32)
        pow_67 = hidden_states_330.pow(2)
        variance_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_198 = variance_66 + 1e-05
        variance_66 = None
        rsqrt_66 = torch.rsqrt(add_198)
        add_198 = None
        hidden_states_331 = hidden_states_330 * rsqrt_66
        hidden_states_330 = rsqrt_66 = None
        to_137 = hidden_states_331.to(torch.float16)
        hidden_states_331 = None
        hidden_states_332 = (
            l_self_modules_model_modules_layers_modules_33_modules_input_layernorm_parameters_weight_
            * to_137
        )
        l_self_modules_model_modules_layers_modules_33_modules_input_layernorm_parameters_weight_ = (
            to_137
        ) = None
        linear_231 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_99 = linear_231.view((1, 19, -1, 128))
        linear_231 = None
        query_states_33 = view_99.transpose(1, 2)
        view_99 = None
        linear_232 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_100 = linear_232.view((1, 19, -1, 128))
        linear_232 = None
        key_states_33 = view_100.transpose(1, 2)
        view_100 = None
        linear_233 = torch._C._nn.linear(
            hidden_states_332,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_332 = l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_101 = linear_233.view((1, 19, -1, 128))
        linear_233 = None
        value_states_33 = view_101.transpose(1, 2)
        view_101 = None
        cos_36 = cos_2.unsqueeze(1)
        sin_36 = sin_2.unsqueeze(1)
        mul_301 = query_states_33 * cos_36
        x1_66 = query_states_33[(Ellipsis, slice(None, 64, None))]
        x2_66 = query_states_33[(Ellipsis, slice(64, None, None))]
        query_states_33 = None
        neg_66 = -x2_66
        x2_66 = None
        cat_67 = torch.cat((neg_66, x1_66), dim=-1)
        neg_66 = x1_66 = None
        mul_302 = cat_67 * sin_36
        cat_67 = None
        q_embed_33 = mul_301 + mul_302
        mul_301 = mul_302 = None
        mul_303 = key_states_33 * cos_36
        cos_36 = None
        x1_67 = key_states_33[(Ellipsis, slice(None, 64, None))]
        x2_67 = key_states_33[(Ellipsis, slice(64, None, None))]
        key_states_33 = None
        neg_67 = -x2_67
        x2_67 = None
        cat_68 = torch.cat((neg_67, x1_67), dim=-1)
        neg_67 = x1_67 = None
        mul_304 = cat_68 * sin_36
        cat_68 = sin_36 = None
        k_embed_33 = mul_303 + mul_304
        mul_303 = mul_304 = None
        getitem_237 = k_embed_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_33 = None
        hidden_states_333 = getitem_237.expand(1, 8, 5, 19, 128)
        getitem_237 = None
        key_66 = hidden_states_333.reshape(1, 40, 19, 128)
        hidden_states_333 = None
        getitem_238 = value_states_33[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_33 = None
        hidden_states_334 = getitem_238.expand(1, 8, 5, 19, 128)
        getitem_238 = None
        value_66 = hidden_states_334.reshape(1, 40, 19, 128)
        hidden_states_334 = None
        attention_mask_34 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_101 = attn_output_133.reshape(1, 19, -1)
        attn_output_133 = None
        attn_output_134 = reshape_101.contiguous()
        reshape_101 = None
        attn_output_135 = torch._C._nn.linear(
            attn_output_134,
            l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_134 = l_self_modules_model_modules_layers_modules_33_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_335 = hidden_states_329 + attn_output_135
        hidden_states_329 = attn_output_135 = None
        hidden_states_336 = hidden_states_335.to(torch.float32)
        pow_68 = hidden_states_336.pow(2)
        variance_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_202 = variance_67 + 1e-05
        variance_67 = None
        rsqrt_67 = torch.rsqrt(add_202)
        add_202 = None
        hidden_states_337 = hidden_states_336 * rsqrt_67
        hidden_states_336 = rsqrt_67 = None
        to_139 = hidden_states_337.to(torch.float16)
        hidden_states_337 = None
        hidden_states_338 = (
            l_self_modules_model_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_
            * to_139
        )
        l_self_modules_model_modules_layers_modules_33_modules_post_attention_layernorm_parameters_weight_ = (
            to_139
        ) = None
        linear_235 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_33 = torch.nn.functional.silu(linear_235, inplace=False)
        linear_235 = None
        linear_236 = torch._C._nn.linear(
            hidden_states_338,
            l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_338 = l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_307 = silu_33 * linear_236
        silu_33 = linear_236 = None
        down_proj_33 = torch._C._nn.linear(
            mul_307,
            l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_307 = l_self_modules_model_modules_layers_modules_33_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_339 = hidden_states_335 + down_proj_33
        hidden_states_335 = down_proj_33 = None
        hidden_states_340 = hidden_states_339.to(torch.float32)
        pow_69 = hidden_states_340.pow(2)
        variance_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_204 = variance_68 + 1e-05
        variance_68 = None
        rsqrt_68 = torch.rsqrt(add_204)
        add_204 = None
        hidden_states_341 = hidden_states_340 * rsqrt_68
        hidden_states_340 = rsqrt_68 = None
        to_141 = hidden_states_341.to(torch.float16)
        hidden_states_341 = None
        hidden_states_342 = (
            l_self_modules_model_modules_layers_modules_34_modules_input_layernorm_parameters_weight_
            * to_141
        )
        l_self_modules_model_modules_layers_modules_34_modules_input_layernorm_parameters_weight_ = (
            to_141
        ) = None
        linear_238 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_102 = linear_238.view((1, 19, -1, 128))
        linear_238 = None
        query_states_34 = view_102.transpose(1, 2)
        view_102 = None
        linear_239 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_103 = linear_239.view((1, 19, -1, 128))
        linear_239 = None
        key_states_34 = view_103.transpose(1, 2)
        view_103 = None
        linear_240 = torch._C._nn.linear(
            hidden_states_342,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_342 = l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_104 = linear_240.view((1, 19, -1, 128))
        linear_240 = None
        value_states_34 = view_104.transpose(1, 2)
        view_104 = None
        cos_37 = cos_2.unsqueeze(1)
        sin_37 = sin_2.unsqueeze(1)
        mul_310 = query_states_34 * cos_37
        x1_68 = query_states_34[(Ellipsis, slice(None, 64, None))]
        x2_68 = query_states_34[(Ellipsis, slice(64, None, None))]
        query_states_34 = None
        neg_68 = -x2_68
        x2_68 = None
        cat_69 = torch.cat((neg_68, x1_68), dim=-1)
        neg_68 = x1_68 = None
        mul_311 = cat_69 * sin_37
        cat_69 = None
        q_embed_34 = mul_310 + mul_311
        mul_310 = mul_311 = None
        mul_312 = key_states_34 * cos_37
        cos_37 = None
        x1_69 = key_states_34[(Ellipsis, slice(None, 64, None))]
        x2_69 = key_states_34[(Ellipsis, slice(64, None, None))]
        key_states_34 = None
        neg_69 = -x2_69
        x2_69 = None
        cat_70 = torch.cat((neg_69, x1_69), dim=-1)
        neg_69 = x1_69 = None
        mul_313 = cat_70 * sin_37
        cat_70 = sin_37 = None
        k_embed_34 = mul_312 + mul_313
        mul_312 = mul_313 = None
        getitem_244 = k_embed_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_34 = None
        hidden_states_343 = getitem_244.expand(1, 8, 5, 19, 128)
        getitem_244 = None
        key_68 = hidden_states_343.reshape(1, 40, 19, 128)
        hidden_states_343 = None
        getitem_245 = value_states_34[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_34 = None
        hidden_states_344 = getitem_245.expand(1, 8, 5, 19, 128)
        getitem_245 = None
        value_68 = hidden_states_344.reshape(1, 40, 19, 128)
        hidden_states_344 = None
        attention_mask_35 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
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
        reshape_104 = attn_output_137.reshape(1, 19, -1)
        attn_output_137 = None
        attn_output_138 = reshape_104.contiguous()
        reshape_104 = None
        attn_output_139 = torch._C._nn.linear(
            attn_output_138,
            l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_138 = l_self_modules_model_modules_layers_modules_34_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_345 = hidden_states_339 + attn_output_139
        hidden_states_339 = attn_output_139 = None
        hidden_states_346 = hidden_states_345.to(torch.float32)
        pow_70 = hidden_states_346.pow(2)
        variance_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_208 = variance_69 + 1e-05
        variance_69 = None
        rsqrt_69 = torch.rsqrt(add_208)
        add_208 = None
        hidden_states_347 = hidden_states_346 * rsqrt_69
        hidden_states_346 = rsqrt_69 = None
        to_143 = hidden_states_347.to(torch.float16)
        hidden_states_347 = None
        hidden_states_348 = (
            l_self_modules_model_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_
            * to_143
        )
        l_self_modules_model_modules_layers_modules_34_modules_post_attention_layernorm_parameters_weight_ = (
            to_143
        ) = None
        linear_242 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_34 = torch.nn.functional.silu(linear_242, inplace=False)
        linear_242 = None
        linear_243 = torch._C._nn.linear(
            hidden_states_348,
            l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_348 = l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_316 = silu_34 * linear_243
        silu_34 = linear_243 = None
        down_proj_34 = torch._C._nn.linear(
            mul_316,
            l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_316 = l_self_modules_model_modules_layers_modules_34_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_349 = hidden_states_345 + down_proj_34
        hidden_states_345 = down_proj_34 = None
        hidden_states_350 = hidden_states_349.to(torch.float32)
        pow_71 = hidden_states_350.pow(2)
        variance_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_210 = variance_70 + 1e-05
        variance_70 = None
        rsqrt_70 = torch.rsqrt(add_210)
        add_210 = None
        hidden_states_351 = hidden_states_350 * rsqrt_70
        hidden_states_350 = rsqrt_70 = None
        to_145 = hidden_states_351.to(torch.float16)
        hidden_states_351 = None
        hidden_states_352 = (
            l_self_modules_model_modules_layers_modules_35_modules_input_layernorm_parameters_weight_
            * to_145
        )
        l_self_modules_model_modules_layers_modules_35_modules_input_layernorm_parameters_weight_ = (
            to_145
        ) = None
        linear_245 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_105 = linear_245.view((1, 19, -1, 128))
        linear_245 = None
        query_states_35 = view_105.transpose(1, 2)
        view_105 = None
        linear_246 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_106 = linear_246.view((1, 19, -1, 128))
        linear_246 = None
        key_states_35 = view_106.transpose(1, 2)
        view_106 = None
        linear_247 = torch._C._nn.linear(
            hidden_states_352,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_352 = l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_107 = linear_247.view((1, 19, -1, 128))
        linear_247 = None
        value_states_35 = view_107.transpose(1, 2)
        view_107 = None
        cos_38 = cos_2.unsqueeze(1)
        sin_38 = sin_2.unsqueeze(1)
        mul_319 = query_states_35 * cos_38
        x1_70 = query_states_35[(Ellipsis, slice(None, 64, None))]
        x2_70 = query_states_35[(Ellipsis, slice(64, None, None))]
        query_states_35 = None
        neg_70 = -x2_70
        x2_70 = None
        cat_71 = torch.cat((neg_70, x1_70), dim=-1)
        neg_70 = x1_70 = None
        mul_320 = cat_71 * sin_38
        cat_71 = None
        q_embed_35 = mul_319 + mul_320
        mul_319 = mul_320 = None
        mul_321 = key_states_35 * cos_38
        cos_38 = None
        x1_71 = key_states_35[(Ellipsis, slice(None, 64, None))]
        x2_71 = key_states_35[(Ellipsis, slice(64, None, None))]
        key_states_35 = None
        neg_71 = -x2_71
        x2_71 = None
        cat_72 = torch.cat((neg_71, x1_71), dim=-1)
        neg_71 = x1_71 = None
        mul_322 = cat_72 * sin_38
        cat_72 = sin_38 = None
        k_embed_35 = mul_321 + mul_322
        mul_321 = mul_322 = None
        getitem_251 = k_embed_35[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_35 = None
        hidden_states_353 = getitem_251.expand(1, 8, 5, 19, 128)
        getitem_251 = None
        key_70 = hidden_states_353.reshape(1, 40, 19, 128)
        hidden_states_353 = None
        getitem_252 = value_states_35[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_35 = None
        hidden_states_354 = getitem_252.expand(1, 8, 5, 19, 128)
        getitem_252 = None
        value_70 = hidden_states_354.reshape(1, 40, 19, 128)
        hidden_states_354 = None
        attention_mask_36 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
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
        reshape_107 = attn_output_141.reshape(1, 19, -1)
        attn_output_141 = None
        attn_output_142 = reshape_107.contiguous()
        reshape_107 = None
        attn_output_143 = torch._C._nn.linear(
            attn_output_142,
            l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_142 = l_self_modules_model_modules_layers_modules_35_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_355 = hidden_states_349 + attn_output_143
        hidden_states_349 = attn_output_143 = None
        hidden_states_356 = hidden_states_355.to(torch.float32)
        pow_72 = hidden_states_356.pow(2)
        variance_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_214 = variance_71 + 1e-05
        variance_71 = None
        rsqrt_71 = torch.rsqrt(add_214)
        add_214 = None
        hidden_states_357 = hidden_states_356 * rsqrt_71
        hidden_states_356 = rsqrt_71 = None
        to_147 = hidden_states_357.to(torch.float16)
        hidden_states_357 = None
        hidden_states_358 = (
            l_self_modules_model_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_
            * to_147
        )
        l_self_modules_model_modules_layers_modules_35_modules_post_attention_layernorm_parameters_weight_ = (
            to_147
        ) = None
        linear_249 = torch._C._nn.linear(
            hidden_states_358,
            l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_35 = torch.nn.functional.silu(linear_249, inplace=False)
        linear_249 = None
        linear_250 = torch._C._nn.linear(
            hidden_states_358,
            l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_358 = l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_325 = silu_35 * linear_250
        silu_35 = linear_250 = None
        down_proj_35 = torch._C._nn.linear(
            mul_325,
            l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_325 = l_self_modules_model_modules_layers_modules_35_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_359 = hidden_states_355 + down_proj_35
        hidden_states_355 = down_proj_35 = None
        hidden_states_360 = hidden_states_359.to(torch.float32)
        pow_73 = hidden_states_360.pow(2)
        variance_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_216 = variance_72 + 1e-05
        variance_72 = None
        rsqrt_72 = torch.rsqrt(add_216)
        add_216 = None
        hidden_states_361 = hidden_states_360 * rsqrt_72
        hidden_states_360 = rsqrt_72 = None
        to_149 = hidden_states_361.to(torch.float16)
        hidden_states_361 = None
        hidden_states_362 = (
            l_self_modules_model_modules_layers_modules_36_modules_input_layernorm_parameters_weight_
            * to_149
        )
        l_self_modules_model_modules_layers_modules_36_modules_input_layernorm_parameters_weight_ = (
            to_149
        ) = None
        linear_252 = torch._C._nn.linear(
            hidden_states_362,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_108 = linear_252.view((1, 19, -1, 128))
        linear_252 = None
        query_states_36 = view_108.transpose(1, 2)
        view_108 = None
        linear_253 = torch._C._nn.linear(
            hidden_states_362,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_109 = linear_253.view((1, 19, -1, 128))
        linear_253 = None
        key_states_36 = view_109.transpose(1, 2)
        view_109 = None
        linear_254 = torch._C._nn.linear(
            hidden_states_362,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_362 = l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_110 = linear_254.view((1, 19, -1, 128))
        linear_254 = None
        value_states_36 = view_110.transpose(1, 2)
        view_110 = None
        cos_39 = cos_2.unsqueeze(1)
        sin_39 = sin_2.unsqueeze(1)
        mul_328 = query_states_36 * cos_39
        x1_72 = query_states_36[(Ellipsis, slice(None, 64, None))]
        x2_72 = query_states_36[(Ellipsis, slice(64, None, None))]
        query_states_36 = None
        neg_72 = -x2_72
        x2_72 = None
        cat_73 = torch.cat((neg_72, x1_72), dim=-1)
        neg_72 = x1_72 = None
        mul_329 = cat_73 * sin_39
        cat_73 = None
        q_embed_36 = mul_328 + mul_329
        mul_328 = mul_329 = None
        mul_330 = key_states_36 * cos_39
        cos_39 = None
        x1_73 = key_states_36[(Ellipsis, slice(None, 64, None))]
        x2_73 = key_states_36[(Ellipsis, slice(64, None, None))]
        key_states_36 = None
        neg_73 = -x2_73
        x2_73 = None
        cat_74 = torch.cat((neg_73, x1_73), dim=-1)
        neg_73 = x1_73 = None
        mul_331 = cat_74 * sin_39
        cat_74 = sin_39 = None
        k_embed_36 = mul_330 + mul_331
        mul_330 = mul_331 = None
        getitem_258 = k_embed_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_36 = None
        hidden_states_363 = getitem_258.expand(1, 8, 5, 19, 128)
        getitem_258 = None
        key_72 = hidden_states_363.reshape(1, 40, 19, 128)
        hidden_states_363 = None
        getitem_259 = value_states_36[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_36 = None
        hidden_states_364 = getitem_259.expand(1, 8, 5, 19, 128)
        getitem_259 = None
        value_72 = hidden_states_364.reshape(1, 40, 19, 128)
        hidden_states_364 = None
        attention_mask_37 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_36 = q_embed_36.contiguous()
        q_embed_36 = None
        key_73 = key_72.contiguous()
        key_72 = None
        value_73 = value_72.contiguous()
        value_72 = None
        attn_output_144 = torch._C._nn.scaled_dot_product_attention(
            query_36,
            key_73,
            value_73,
            attn_mask=attention_mask_37,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_36 = key_73 = value_73 = attention_mask_37 = None
        transpose_148 = attn_output_144.transpose(1, 2)
        attn_output_144 = None
        attn_output_145 = transpose_148.contiguous()
        transpose_148 = None
        reshape_110 = attn_output_145.reshape(1, 19, -1)
        attn_output_145 = None
        attn_output_146 = reshape_110.contiguous()
        reshape_110 = None
        attn_output_147 = torch._C._nn.linear(
            attn_output_146,
            l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_146 = l_self_modules_model_modules_layers_modules_36_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_365 = hidden_states_359 + attn_output_147
        hidden_states_359 = attn_output_147 = None
        hidden_states_366 = hidden_states_365.to(torch.float32)
        pow_74 = hidden_states_366.pow(2)
        variance_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_220 = variance_73 + 1e-05
        variance_73 = None
        rsqrt_73 = torch.rsqrt(add_220)
        add_220 = None
        hidden_states_367 = hidden_states_366 * rsqrt_73
        hidden_states_366 = rsqrt_73 = None
        to_151 = hidden_states_367.to(torch.float16)
        hidden_states_367 = None
        hidden_states_368 = (
            l_self_modules_model_modules_layers_modules_36_modules_post_attention_layernorm_parameters_weight_
            * to_151
        )
        l_self_modules_model_modules_layers_modules_36_modules_post_attention_layernorm_parameters_weight_ = (
            to_151
        ) = None
        linear_256 = torch._C._nn.linear(
            hidden_states_368,
            l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_36 = torch.nn.functional.silu(linear_256, inplace=False)
        linear_256 = None
        linear_257 = torch._C._nn.linear(
            hidden_states_368,
            l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_368 = l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_334 = silu_36 * linear_257
        silu_36 = linear_257 = None
        down_proj_36 = torch._C._nn.linear(
            mul_334,
            l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_334 = l_self_modules_model_modules_layers_modules_36_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_369 = hidden_states_365 + down_proj_36
        hidden_states_365 = down_proj_36 = None
        hidden_states_370 = hidden_states_369.to(torch.float32)
        pow_75 = hidden_states_370.pow(2)
        variance_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_222 = variance_74 + 1e-05
        variance_74 = None
        rsqrt_74 = torch.rsqrt(add_222)
        add_222 = None
        hidden_states_371 = hidden_states_370 * rsqrt_74
        hidden_states_370 = rsqrt_74 = None
        to_153 = hidden_states_371.to(torch.float16)
        hidden_states_371 = None
        hidden_states_372 = (
            l_self_modules_model_modules_layers_modules_37_modules_input_layernorm_parameters_weight_
            * to_153
        )
        l_self_modules_model_modules_layers_modules_37_modules_input_layernorm_parameters_weight_ = (
            to_153
        ) = None
        linear_259 = torch._C._nn.linear(
            hidden_states_372,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_111 = linear_259.view((1, 19, -1, 128))
        linear_259 = None
        query_states_37 = view_111.transpose(1, 2)
        view_111 = None
        linear_260 = torch._C._nn.linear(
            hidden_states_372,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_112 = linear_260.view((1, 19, -1, 128))
        linear_260 = None
        key_states_37 = view_112.transpose(1, 2)
        view_112 = None
        linear_261 = torch._C._nn.linear(
            hidden_states_372,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_372 = l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_113 = linear_261.view((1, 19, -1, 128))
        linear_261 = None
        value_states_37 = view_113.transpose(1, 2)
        view_113 = None
        cos_40 = cos_2.unsqueeze(1)
        sin_40 = sin_2.unsqueeze(1)
        mul_337 = query_states_37 * cos_40
        x1_74 = query_states_37[(Ellipsis, slice(None, 64, None))]
        x2_74 = query_states_37[(Ellipsis, slice(64, None, None))]
        query_states_37 = None
        neg_74 = -x2_74
        x2_74 = None
        cat_75 = torch.cat((neg_74, x1_74), dim=-1)
        neg_74 = x1_74 = None
        mul_338 = cat_75 * sin_40
        cat_75 = None
        q_embed_37 = mul_337 + mul_338
        mul_337 = mul_338 = None
        mul_339 = key_states_37 * cos_40
        cos_40 = None
        x1_75 = key_states_37[(Ellipsis, slice(None, 64, None))]
        x2_75 = key_states_37[(Ellipsis, slice(64, None, None))]
        key_states_37 = None
        neg_75 = -x2_75
        x2_75 = None
        cat_76 = torch.cat((neg_75, x1_75), dim=-1)
        neg_75 = x1_75 = None
        mul_340 = cat_76 * sin_40
        cat_76 = sin_40 = None
        k_embed_37 = mul_339 + mul_340
        mul_339 = mul_340 = None
        getitem_265 = k_embed_37[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_37 = None
        hidden_states_373 = getitem_265.expand(1, 8, 5, 19, 128)
        getitem_265 = None
        key_74 = hidden_states_373.reshape(1, 40, 19, 128)
        hidden_states_373 = None
        getitem_266 = value_states_37[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_37 = None
        hidden_states_374 = getitem_266.expand(1, 8, 5, 19, 128)
        getitem_266 = None
        value_74 = hidden_states_374.reshape(1, 40, 19, 128)
        hidden_states_374 = None
        attention_mask_38 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_37 = q_embed_37.contiguous()
        q_embed_37 = None
        key_75 = key_74.contiguous()
        key_74 = None
        value_75 = value_74.contiguous()
        value_74 = None
        attn_output_148 = torch._C._nn.scaled_dot_product_attention(
            query_37,
            key_75,
            value_75,
            attn_mask=attention_mask_38,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_37 = key_75 = value_75 = attention_mask_38 = None
        transpose_152 = attn_output_148.transpose(1, 2)
        attn_output_148 = None
        attn_output_149 = transpose_152.contiguous()
        transpose_152 = None
        reshape_113 = attn_output_149.reshape(1, 19, -1)
        attn_output_149 = None
        attn_output_150 = reshape_113.contiguous()
        reshape_113 = None
        attn_output_151 = torch._C._nn.linear(
            attn_output_150,
            l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_150 = l_self_modules_model_modules_layers_modules_37_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_375 = hidden_states_369 + attn_output_151
        hidden_states_369 = attn_output_151 = None
        hidden_states_376 = hidden_states_375.to(torch.float32)
        pow_76 = hidden_states_376.pow(2)
        variance_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_226 = variance_75 + 1e-05
        variance_75 = None
        rsqrt_75 = torch.rsqrt(add_226)
        add_226 = None
        hidden_states_377 = hidden_states_376 * rsqrt_75
        hidden_states_376 = rsqrt_75 = None
        to_155 = hidden_states_377.to(torch.float16)
        hidden_states_377 = None
        hidden_states_378 = (
            l_self_modules_model_modules_layers_modules_37_modules_post_attention_layernorm_parameters_weight_
            * to_155
        )
        l_self_modules_model_modules_layers_modules_37_modules_post_attention_layernorm_parameters_weight_ = (
            to_155
        ) = None
        linear_263 = torch._C._nn.linear(
            hidden_states_378,
            l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_37 = torch.nn.functional.silu(linear_263, inplace=False)
        linear_263 = None
        linear_264 = torch._C._nn.linear(
            hidden_states_378,
            l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_378 = l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_343 = silu_37 * linear_264
        silu_37 = linear_264 = None
        down_proj_37 = torch._C._nn.linear(
            mul_343,
            l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_343 = l_self_modules_model_modules_layers_modules_37_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_379 = hidden_states_375 + down_proj_37
        hidden_states_375 = down_proj_37 = None
        hidden_states_380 = hidden_states_379.to(torch.float32)
        pow_77 = hidden_states_380.pow(2)
        variance_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_228 = variance_76 + 1e-05
        variance_76 = None
        rsqrt_76 = torch.rsqrt(add_228)
        add_228 = None
        hidden_states_381 = hidden_states_380 * rsqrt_76
        hidden_states_380 = rsqrt_76 = None
        to_157 = hidden_states_381.to(torch.float16)
        hidden_states_381 = None
        hidden_states_382 = (
            l_self_modules_model_modules_layers_modules_38_modules_input_layernorm_parameters_weight_
            * to_157
        )
        l_self_modules_model_modules_layers_modules_38_modules_input_layernorm_parameters_weight_ = (
            to_157
        ) = None
        linear_266 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_114 = linear_266.view((1, 19, -1, 128))
        linear_266 = None
        query_states_38 = view_114.transpose(1, 2)
        view_114 = None
        linear_267 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_115 = linear_267.view((1, 19, -1, 128))
        linear_267 = None
        key_states_38 = view_115.transpose(1, 2)
        view_115 = None
        linear_268 = torch._C._nn.linear(
            hidden_states_382,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_382 = l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_116 = linear_268.view((1, 19, -1, 128))
        linear_268 = None
        value_states_38 = view_116.transpose(1, 2)
        view_116 = None
        cos_41 = cos_2.unsqueeze(1)
        sin_41 = sin_2.unsqueeze(1)
        mul_346 = query_states_38 * cos_41
        x1_76 = query_states_38[(Ellipsis, slice(None, 64, None))]
        x2_76 = query_states_38[(Ellipsis, slice(64, None, None))]
        query_states_38 = None
        neg_76 = -x2_76
        x2_76 = None
        cat_77 = torch.cat((neg_76, x1_76), dim=-1)
        neg_76 = x1_76 = None
        mul_347 = cat_77 * sin_41
        cat_77 = None
        q_embed_38 = mul_346 + mul_347
        mul_346 = mul_347 = None
        mul_348 = key_states_38 * cos_41
        cos_41 = None
        x1_77 = key_states_38[(Ellipsis, slice(None, 64, None))]
        x2_77 = key_states_38[(Ellipsis, slice(64, None, None))]
        key_states_38 = None
        neg_77 = -x2_77
        x2_77 = None
        cat_78 = torch.cat((neg_77, x1_77), dim=-1)
        neg_77 = x1_77 = None
        mul_349 = cat_78 * sin_41
        cat_78 = sin_41 = None
        k_embed_38 = mul_348 + mul_349
        mul_348 = mul_349 = None
        getitem_272 = k_embed_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_38 = None
        hidden_states_383 = getitem_272.expand(1, 8, 5, 19, 128)
        getitem_272 = None
        key_76 = hidden_states_383.reshape(1, 40, 19, 128)
        hidden_states_383 = None
        getitem_273 = value_states_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_38 = None
        hidden_states_384 = getitem_273.expand(1, 8, 5, 19, 128)
        getitem_273 = None
        value_76 = hidden_states_384.reshape(1, 40, 19, 128)
        hidden_states_384 = None
        attention_mask_39 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_38 = q_embed_38.contiguous()
        q_embed_38 = None
        key_77 = key_76.contiguous()
        key_76 = None
        value_77 = value_76.contiguous()
        value_76 = None
        attn_output_152 = torch._C._nn.scaled_dot_product_attention(
            query_38,
            key_77,
            value_77,
            attn_mask=attention_mask_39,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_38 = key_77 = value_77 = attention_mask_39 = None
        transpose_156 = attn_output_152.transpose(1, 2)
        attn_output_152 = None
        attn_output_153 = transpose_156.contiguous()
        transpose_156 = None
        reshape_116 = attn_output_153.reshape(1, 19, -1)
        attn_output_153 = None
        attn_output_154 = reshape_116.contiguous()
        reshape_116 = None
        attn_output_155 = torch._C._nn.linear(
            attn_output_154,
            l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_154 = l_self_modules_model_modules_layers_modules_38_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_385 = hidden_states_379 + attn_output_155
        hidden_states_379 = attn_output_155 = None
        hidden_states_386 = hidden_states_385.to(torch.float32)
        pow_78 = hidden_states_386.pow(2)
        variance_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_232 = variance_77 + 1e-05
        variance_77 = None
        rsqrt_77 = torch.rsqrt(add_232)
        add_232 = None
        hidden_states_387 = hidden_states_386 * rsqrt_77
        hidden_states_386 = rsqrt_77 = None
        to_159 = hidden_states_387.to(torch.float16)
        hidden_states_387 = None
        hidden_states_388 = (
            l_self_modules_model_modules_layers_modules_38_modules_post_attention_layernorm_parameters_weight_
            * to_159
        )
        l_self_modules_model_modules_layers_modules_38_modules_post_attention_layernorm_parameters_weight_ = (
            to_159
        ) = None
        linear_270 = torch._C._nn.linear(
            hidden_states_388,
            l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_38 = torch.nn.functional.silu(linear_270, inplace=False)
        linear_270 = None
        linear_271 = torch._C._nn.linear(
            hidden_states_388,
            l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_388 = l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_352 = silu_38 * linear_271
        silu_38 = linear_271 = None
        down_proj_38 = torch._C._nn.linear(
            mul_352,
            l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_352 = l_self_modules_model_modules_layers_modules_38_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_389 = hidden_states_385 + down_proj_38
        hidden_states_385 = down_proj_38 = None
        hidden_states_390 = hidden_states_389.to(torch.float32)
        pow_79 = hidden_states_390.pow(2)
        variance_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        add_234 = variance_78 + 1e-05
        variance_78 = None
        rsqrt_78 = torch.rsqrt(add_234)
        add_234 = None
        hidden_states_391 = hidden_states_390 * rsqrt_78
        hidden_states_390 = rsqrt_78 = None
        to_161 = hidden_states_391.to(torch.float16)
        hidden_states_391 = None
        hidden_states_392 = (
            l_self_modules_model_modules_layers_modules_39_modules_input_layernorm_parameters_weight_
            * to_161
        )
        l_self_modules_model_modules_layers_modules_39_modules_input_layernorm_parameters_weight_ = (
            to_161
        ) = None
        linear_273 = torch._C._nn.linear(
            hidden_states_392,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_117 = linear_273.view((1, 19, -1, 128))
        linear_273 = None
        query_states_39 = view_117.transpose(1, 2)
        view_117 = None
        linear_274 = torch._C._nn.linear(
            hidden_states_392,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_118 = linear_274.view((1, 19, -1, 128))
        linear_274 = None
        key_states_39 = view_118.transpose(1, 2)
        view_118 = None
        linear_275 = torch._C._nn.linear(
            hidden_states_392,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_392 = l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_119 = linear_275.view((1, 19, -1, 128))
        linear_275 = None
        value_states_39 = view_119.transpose(1, 2)
        view_119 = None
        cos_42 = cos_2.unsqueeze(1)
        sin_42 = sin_2.unsqueeze(1)
        mul_355 = query_states_39 * cos_42
        x1_78 = query_states_39[(Ellipsis, slice(None, 64, None))]
        x2_78 = query_states_39[(Ellipsis, slice(64, None, None))]
        query_states_39 = None
        neg_78 = -x2_78
        x2_78 = None
        cat_79 = torch.cat((neg_78, x1_78), dim=-1)
        neg_78 = x1_78 = None
        mul_356 = cat_79 * sin_42
        cat_79 = None
        q_embed_39 = mul_355 + mul_356
        mul_355 = mul_356 = None
        mul_357 = key_states_39 * cos_42
        cos_42 = None
        x1_79 = key_states_39[(Ellipsis, slice(None, 64, None))]
        x2_79 = key_states_39[(Ellipsis, slice(64, None, None))]
        key_states_39 = None
        neg_79 = -x2_79
        x2_79 = None
        cat_80 = torch.cat((neg_79, x1_79), dim=-1)
        neg_79 = x1_79 = None
        mul_358 = cat_80 * sin_42
        cat_80 = sin_42 = None
        k_embed_39 = mul_357 + mul_358
        mul_357 = mul_358 = None
        getitem_279 = k_embed_39[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_39 = None
        hidden_states_393 = getitem_279.expand(1, 8, 5, 19, 128)
        getitem_279 = None
        key_78 = hidden_states_393.reshape(1, 40, 19, 128)
        hidden_states_393 = None
        getitem_280 = value_states_39[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_39 = None
        hidden_states_394 = getitem_280.expand(1, 8, 5, 19, 128)
        getitem_280 = None
        value_78 = hidden_states_394.reshape(1, 40, 19, 128)
        hidden_states_394 = None
        attention_mask_40 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_39 = q_embed_39.contiguous()
        q_embed_39 = None
        key_79 = key_78.contiguous()
        key_78 = None
        value_79 = value_78.contiguous()
        value_78 = None
        attn_output_156 = torch._C._nn.scaled_dot_product_attention(
            query_39,
            key_79,
            value_79,
            attn_mask=attention_mask_40,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_39 = key_79 = value_79 = attention_mask_40 = None
        transpose_160 = attn_output_156.transpose(1, 2)
        attn_output_156 = None
        attn_output_157 = transpose_160.contiguous()
        transpose_160 = None
        reshape_119 = attn_output_157.reshape(1, 19, -1)
        attn_output_157 = None
        attn_output_158 = reshape_119.contiguous()
        reshape_119 = None
        attn_output_159 = torch._C._nn.linear(
            attn_output_158,
            l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_158 = l_self_modules_model_modules_layers_modules_39_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_395 = hidden_states_389 + attn_output_159
        hidden_states_389 = attn_output_159 = None
        hidden_states_396 = hidden_states_395.to(torch.float32)
        pow_80 = hidden_states_396.pow(2)
        variance_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_238 = variance_79 + 1e-05
        variance_79 = None
        rsqrt_79 = torch.rsqrt(add_238)
        add_238 = None
        hidden_states_397 = hidden_states_396 * rsqrt_79
        hidden_states_396 = rsqrt_79 = None
        to_163 = hidden_states_397.to(torch.float16)
        hidden_states_397 = None
        hidden_states_398 = (
            l_self_modules_model_modules_layers_modules_39_modules_post_attention_layernorm_parameters_weight_
            * to_163
        )
        l_self_modules_model_modules_layers_modules_39_modules_post_attention_layernorm_parameters_weight_ = (
            to_163
        ) = None
        linear_277 = torch._C._nn.linear(
            hidden_states_398,
            l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_39 = torch.nn.functional.silu(linear_277, inplace=False)
        linear_277 = None
        linear_278 = torch._C._nn.linear(
            hidden_states_398,
            l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_398 = l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_361 = silu_39 * linear_278
        silu_39 = linear_278 = None
        down_proj_39 = torch._C._nn.linear(
            mul_361,
            l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_361 = l_self_modules_model_modules_layers_modules_39_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_399 = hidden_states_395 + down_proj_39
        hidden_states_395 = down_proj_39 = None
        hidden_states_400 = hidden_states_399.to(torch.float32)
        pow_81 = hidden_states_400.pow(2)
        variance_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_240 = variance_80 + 1e-05
        variance_80 = None
        rsqrt_80 = torch.rsqrt(add_240)
        add_240 = None
        hidden_states_401 = hidden_states_400 * rsqrt_80
        hidden_states_400 = rsqrt_80 = None
        to_165 = hidden_states_401.to(torch.float16)
        hidden_states_401 = None
        hidden_states_402 = (
            l_self_modules_model_modules_layers_modules_40_modules_input_layernorm_parameters_weight_
            * to_165
        )
        l_self_modules_model_modules_layers_modules_40_modules_input_layernorm_parameters_weight_ = (
            to_165
        ) = None
        linear_280 = torch._C._nn.linear(
            hidden_states_402,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_120 = linear_280.view((1, 19, -1, 128))
        linear_280 = None
        query_states_40 = view_120.transpose(1, 2)
        view_120 = None
        linear_281 = torch._C._nn.linear(
            hidden_states_402,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_121 = linear_281.view((1, 19, -1, 128))
        linear_281 = None
        key_states_40 = view_121.transpose(1, 2)
        view_121 = None
        linear_282 = torch._C._nn.linear(
            hidden_states_402,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_402 = l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_122 = linear_282.view((1, 19, -1, 128))
        linear_282 = None
        value_states_40 = view_122.transpose(1, 2)
        view_122 = None
        cos_43 = cos_2.unsqueeze(1)
        sin_43 = sin_2.unsqueeze(1)
        mul_364 = query_states_40 * cos_43
        x1_80 = query_states_40[(Ellipsis, slice(None, 64, None))]
        x2_80 = query_states_40[(Ellipsis, slice(64, None, None))]
        query_states_40 = None
        neg_80 = -x2_80
        x2_80 = None
        cat_81 = torch.cat((neg_80, x1_80), dim=-1)
        neg_80 = x1_80 = None
        mul_365 = cat_81 * sin_43
        cat_81 = None
        q_embed_40 = mul_364 + mul_365
        mul_364 = mul_365 = None
        mul_366 = key_states_40 * cos_43
        cos_43 = None
        x1_81 = key_states_40[(Ellipsis, slice(None, 64, None))]
        x2_81 = key_states_40[(Ellipsis, slice(64, None, None))]
        key_states_40 = None
        neg_81 = -x2_81
        x2_81 = None
        cat_82 = torch.cat((neg_81, x1_81), dim=-1)
        neg_81 = x1_81 = None
        mul_367 = cat_82 * sin_43
        cat_82 = sin_43 = None
        k_embed_40 = mul_366 + mul_367
        mul_366 = mul_367 = None
        getitem_286 = k_embed_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_40 = None
        hidden_states_403 = getitem_286.expand(1, 8, 5, 19, 128)
        getitem_286 = None
        key_80 = hidden_states_403.reshape(1, 40, 19, 128)
        hidden_states_403 = None
        getitem_287 = value_states_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_40 = None
        hidden_states_404 = getitem_287.expand(1, 8, 5, 19, 128)
        getitem_287 = None
        value_80 = hidden_states_404.reshape(1, 40, 19, 128)
        hidden_states_404 = None
        attention_mask_41 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_40 = q_embed_40.contiguous()
        q_embed_40 = None
        key_81 = key_80.contiguous()
        key_80 = None
        value_81 = value_80.contiguous()
        value_80 = None
        attn_output_160 = torch._C._nn.scaled_dot_product_attention(
            query_40,
            key_81,
            value_81,
            attn_mask=attention_mask_41,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_40 = key_81 = value_81 = attention_mask_41 = None
        transpose_164 = attn_output_160.transpose(1, 2)
        attn_output_160 = None
        attn_output_161 = transpose_164.contiguous()
        transpose_164 = None
        reshape_122 = attn_output_161.reshape(1, 19, -1)
        attn_output_161 = None
        attn_output_162 = reshape_122.contiguous()
        reshape_122 = None
        attn_output_163 = torch._C._nn.linear(
            attn_output_162,
            l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_162 = l_self_modules_model_modules_layers_modules_40_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_405 = hidden_states_399 + attn_output_163
        hidden_states_399 = attn_output_163 = None
        hidden_states_406 = hidden_states_405.to(torch.float32)
        pow_82 = hidden_states_406.pow(2)
        variance_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_244 = variance_81 + 1e-05
        variance_81 = None
        rsqrt_81 = torch.rsqrt(add_244)
        add_244 = None
        hidden_states_407 = hidden_states_406 * rsqrt_81
        hidden_states_406 = rsqrt_81 = None
        to_167 = hidden_states_407.to(torch.float16)
        hidden_states_407 = None
        hidden_states_408 = (
            l_self_modules_model_modules_layers_modules_40_modules_post_attention_layernorm_parameters_weight_
            * to_167
        )
        l_self_modules_model_modules_layers_modules_40_modules_post_attention_layernorm_parameters_weight_ = (
            to_167
        ) = None
        linear_284 = torch._C._nn.linear(
            hidden_states_408,
            l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_40 = torch.nn.functional.silu(linear_284, inplace=False)
        linear_284 = None
        linear_285 = torch._C._nn.linear(
            hidden_states_408,
            l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_408 = l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_370 = silu_40 * linear_285
        silu_40 = linear_285 = None
        down_proj_40 = torch._C._nn.linear(
            mul_370,
            l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_370 = l_self_modules_model_modules_layers_modules_40_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_409 = hidden_states_405 + down_proj_40
        hidden_states_405 = down_proj_40 = None
        hidden_states_410 = hidden_states_409.to(torch.float32)
        pow_83 = hidden_states_410.pow(2)
        variance_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        add_246 = variance_82 + 1e-05
        variance_82 = None
        rsqrt_82 = torch.rsqrt(add_246)
        add_246 = None
        hidden_states_411 = hidden_states_410 * rsqrt_82
        hidden_states_410 = rsqrt_82 = None
        to_169 = hidden_states_411.to(torch.float16)
        hidden_states_411 = None
        hidden_states_412 = (
            l_self_modules_model_modules_layers_modules_41_modules_input_layernorm_parameters_weight_
            * to_169
        )
        l_self_modules_model_modules_layers_modules_41_modules_input_layernorm_parameters_weight_ = (
            to_169
        ) = None
        linear_287 = torch._C._nn.linear(
            hidden_states_412,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_123 = linear_287.view((1, 19, -1, 128))
        linear_287 = None
        query_states_41 = view_123.transpose(1, 2)
        view_123 = None
        linear_288 = torch._C._nn.linear(
            hidden_states_412,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_124 = linear_288.view((1, 19, -1, 128))
        linear_288 = None
        key_states_41 = view_124.transpose(1, 2)
        view_124 = None
        linear_289 = torch._C._nn.linear(
            hidden_states_412,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_412 = l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_125 = linear_289.view((1, 19, -1, 128))
        linear_289 = None
        value_states_41 = view_125.transpose(1, 2)
        view_125 = None
        cos_44 = cos_2.unsqueeze(1)
        sin_44 = sin_2.unsqueeze(1)
        mul_373 = query_states_41 * cos_44
        x1_82 = query_states_41[(Ellipsis, slice(None, 64, None))]
        x2_82 = query_states_41[(Ellipsis, slice(64, None, None))]
        query_states_41 = None
        neg_82 = -x2_82
        x2_82 = None
        cat_83 = torch.cat((neg_82, x1_82), dim=-1)
        neg_82 = x1_82 = None
        mul_374 = cat_83 * sin_44
        cat_83 = None
        q_embed_41 = mul_373 + mul_374
        mul_373 = mul_374 = None
        mul_375 = key_states_41 * cos_44
        cos_44 = None
        x1_83 = key_states_41[(Ellipsis, slice(None, 64, None))]
        x2_83 = key_states_41[(Ellipsis, slice(64, None, None))]
        key_states_41 = None
        neg_83 = -x2_83
        x2_83 = None
        cat_84 = torch.cat((neg_83, x1_83), dim=-1)
        neg_83 = x1_83 = None
        mul_376 = cat_84 * sin_44
        cat_84 = sin_44 = None
        k_embed_41 = mul_375 + mul_376
        mul_375 = mul_376 = None
        getitem_293 = k_embed_41[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_41 = None
        hidden_states_413 = getitem_293.expand(1, 8, 5, 19, 128)
        getitem_293 = None
        key_82 = hidden_states_413.reshape(1, 40, 19, 128)
        hidden_states_413 = None
        getitem_294 = value_states_41[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_41 = None
        hidden_states_414 = getitem_294.expand(1, 8, 5, 19, 128)
        getitem_294 = None
        value_82 = hidden_states_414.reshape(1, 40, 19, 128)
        hidden_states_414 = None
        attention_mask_42 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_41 = q_embed_41.contiguous()
        q_embed_41 = None
        key_83 = key_82.contiguous()
        key_82 = None
        value_83 = value_82.contiguous()
        value_82 = None
        attn_output_164 = torch._C._nn.scaled_dot_product_attention(
            query_41,
            key_83,
            value_83,
            attn_mask=attention_mask_42,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_41 = key_83 = value_83 = attention_mask_42 = None
        transpose_168 = attn_output_164.transpose(1, 2)
        attn_output_164 = None
        attn_output_165 = transpose_168.contiguous()
        transpose_168 = None
        reshape_125 = attn_output_165.reshape(1, 19, -1)
        attn_output_165 = None
        attn_output_166 = reshape_125.contiguous()
        reshape_125 = None
        attn_output_167 = torch._C._nn.linear(
            attn_output_166,
            l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_166 = l_self_modules_model_modules_layers_modules_41_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_415 = hidden_states_409 + attn_output_167
        hidden_states_409 = attn_output_167 = None
        hidden_states_416 = hidden_states_415.to(torch.float32)
        pow_84 = hidden_states_416.pow(2)
        variance_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_250 = variance_83 + 1e-05
        variance_83 = None
        rsqrt_83 = torch.rsqrt(add_250)
        add_250 = None
        hidden_states_417 = hidden_states_416 * rsqrt_83
        hidden_states_416 = rsqrt_83 = None
        to_171 = hidden_states_417.to(torch.float16)
        hidden_states_417 = None
        hidden_states_418 = (
            l_self_modules_model_modules_layers_modules_41_modules_post_attention_layernorm_parameters_weight_
            * to_171
        )
        l_self_modules_model_modules_layers_modules_41_modules_post_attention_layernorm_parameters_weight_ = (
            to_171
        ) = None
        linear_291 = torch._C._nn.linear(
            hidden_states_418,
            l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_41 = torch.nn.functional.silu(linear_291, inplace=False)
        linear_291 = None
        linear_292 = torch._C._nn.linear(
            hidden_states_418,
            l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_418 = l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_379 = silu_41 * linear_292
        silu_41 = linear_292 = None
        down_proj_41 = torch._C._nn.linear(
            mul_379,
            l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_379 = l_self_modules_model_modules_layers_modules_41_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_419 = hidden_states_415 + down_proj_41
        hidden_states_415 = down_proj_41 = None
        hidden_states_420 = hidden_states_419.to(torch.float32)
        pow_85 = hidden_states_420.pow(2)
        variance_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_252 = variance_84 + 1e-05
        variance_84 = None
        rsqrt_84 = torch.rsqrt(add_252)
        add_252 = None
        hidden_states_421 = hidden_states_420 * rsqrt_84
        hidden_states_420 = rsqrt_84 = None
        to_173 = hidden_states_421.to(torch.float16)
        hidden_states_421 = None
        hidden_states_422 = (
            l_self_modules_model_modules_layers_modules_42_modules_input_layernorm_parameters_weight_
            * to_173
        )
        l_self_modules_model_modules_layers_modules_42_modules_input_layernorm_parameters_weight_ = (
            to_173
        ) = None
        linear_294 = torch._C._nn.linear(
            hidden_states_422,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_126 = linear_294.view((1, 19, -1, 128))
        linear_294 = None
        query_states_42 = view_126.transpose(1, 2)
        view_126 = None
        linear_295 = torch._C._nn.linear(
            hidden_states_422,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_127 = linear_295.view((1, 19, -1, 128))
        linear_295 = None
        key_states_42 = view_127.transpose(1, 2)
        view_127 = None
        linear_296 = torch._C._nn.linear(
            hidden_states_422,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_422 = l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_128 = linear_296.view((1, 19, -1, 128))
        linear_296 = None
        value_states_42 = view_128.transpose(1, 2)
        view_128 = None
        cos_45 = cos_2.unsqueeze(1)
        sin_45 = sin_2.unsqueeze(1)
        mul_382 = query_states_42 * cos_45
        x1_84 = query_states_42[(Ellipsis, slice(None, 64, None))]
        x2_84 = query_states_42[(Ellipsis, slice(64, None, None))]
        query_states_42 = None
        neg_84 = -x2_84
        x2_84 = None
        cat_85 = torch.cat((neg_84, x1_84), dim=-1)
        neg_84 = x1_84 = None
        mul_383 = cat_85 * sin_45
        cat_85 = None
        q_embed_42 = mul_382 + mul_383
        mul_382 = mul_383 = None
        mul_384 = key_states_42 * cos_45
        cos_45 = None
        x1_85 = key_states_42[(Ellipsis, slice(None, 64, None))]
        x2_85 = key_states_42[(Ellipsis, slice(64, None, None))]
        key_states_42 = None
        neg_85 = -x2_85
        x2_85 = None
        cat_86 = torch.cat((neg_85, x1_85), dim=-1)
        neg_85 = x1_85 = None
        mul_385 = cat_86 * sin_45
        cat_86 = sin_45 = None
        k_embed_42 = mul_384 + mul_385
        mul_384 = mul_385 = None
        getitem_300 = k_embed_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_42 = None
        hidden_states_423 = getitem_300.expand(1, 8, 5, 19, 128)
        getitem_300 = None
        key_84 = hidden_states_423.reshape(1, 40, 19, 128)
        hidden_states_423 = None
        getitem_301 = value_states_42[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_42 = None
        hidden_states_424 = getitem_301.expand(1, 8, 5, 19, 128)
        getitem_301 = None
        value_84 = hidden_states_424.reshape(1, 40, 19, 128)
        hidden_states_424 = None
        attention_mask_43 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_42 = q_embed_42.contiguous()
        q_embed_42 = None
        key_85 = key_84.contiguous()
        key_84 = None
        value_85 = value_84.contiguous()
        value_84 = None
        attn_output_168 = torch._C._nn.scaled_dot_product_attention(
            query_42,
            key_85,
            value_85,
            attn_mask=attention_mask_43,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_42 = key_85 = value_85 = attention_mask_43 = None
        transpose_172 = attn_output_168.transpose(1, 2)
        attn_output_168 = None
        attn_output_169 = transpose_172.contiguous()
        transpose_172 = None
        reshape_128 = attn_output_169.reshape(1, 19, -1)
        attn_output_169 = None
        attn_output_170 = reshape_128.contiguous()
        reshape_128 = None
        attn_output_171 = torch._C._nn.linear(
            attn_output_170,
            l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_170 = l_self_modules_model_modules_layers_modules_42_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_425 = hidden_states_419 + attn_output_171
        hidden_states_419 = attn_output_171 = None
        hidden_states_426 = hidden_states_425.to(torch.float32)
        pow_86 = hidden_states_426.pow(2)
        variance_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_256 = variance_85 + 1e-05
        variance_85 = None
        rsqrt_85 = torch.rsqrt(add_256)
        add_256 = None
        hidden_states_427 = hidden_states_426 * rsqrt_85
        hidden_states_426 = rsqrt_85 = None
        to_175 = hidden_states_427.to(torch.float16)
        hidden_states_427 = None
        hidden_states_428 = (
            l_self_modules_model_modules_layers_modules_42_modules_post_attention_layernorm_parameters_weight_
            * to_175
        )
        l_self_modules_model_modules_layers_modules_42_modules_post_attention_layernorm_parameters_weight_ = (
            to_175
        ) = None
        linear_298 = torch._C._nn.linear(
            hidden_states_428,
            l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_42 = torch.nn.functional.silu(linear_298, inplace=False)
        linear_298 = None
        linear_299 = torch._C._nn.linear(
            hidden_states_428,
            l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_428 = l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_388 = silu_42 * linear_299
        silu_42 = linear_299 = None
        down_proj_42 = torch._C._nn.linear(
            mul_388,
            l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_388 = l_self_modules_model_modules_layers_modules_42_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_429 = hidden_states_425 + down_proj_42
        hidden_states_425 = down_proj_42 = None
        hidden_states_430 = hidden_states_429.to(torch.float32)
        pow_87 = hidden_states_430.pow(2)
        variance_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        add_258 = variance_86 + 1e-05
        variance_86 = None
        rsqrt_86 = torch.rsqrt(add_258)
        add_258 = None
        hidden_states_431 = hidden_states_430 * rsqrt_86
        hidden_states_430 = rsqrt_86 = None
        to_177 = hidden_states_431.to(torch.float16)
        hidden_states_431 = None
        hidden_states_432 = (
            l_self_modules_model_modules_layers_modules_43_modules_input_layernorm_parameters_weight_
            * to_177
        )
        l_self_modules_model_modules_layers_modules_43_modules_input_layernorm_parameters_weight_ = (
            to_177
        ) = None
        linear_301 = torch._C._nn.linear(
            hidden_states_432,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_129 = linear_301.view((1, 19, -1, 128))
        linear_301 = None
        query_states_43 = view_129.transpose(1, 2)
        view_129 = None
        linear_302 = torch._C._nn.linear(
            hidden_states_432,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_130 = linear_302.view((1, 19, -1, 128))
        linear_302 = None
        key_states_43 = view_130.transpose(1, 2)
        view_130 = None
        linear_303 = torch._C._nn.linear(
            hidden_states_432,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_432 = l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_131 = linear_303.view((1, 19, -1, 128))
        linear_303 = None
        value_states_43 = view_131.transpose(1, 2)
        view_131 = None
        cos_46 = cos_2.unsqueeze(1)
        sin_46 = sin_2.unsqueeze(1)
        mul_391 = query_states_43 * cos_46
        x1_86 = query_states_43[(Ellipsis, slice(None, 64, None))]
        x2_86 = query_states_43[(Ellipsis, slice(64, None, None))]
        query_states_43 = None
        neg_86 = -x2_86
        x2_86 = None
        cat_87 = torch.cat((neg_86, x1_86), dim=-1)
        neg_86 = x1_86 = None
        mul_392 = cat_87 * sin_46
        cat_87 = None
        q_embed_43 = mul_391 + mul_392
        mul_391 = mul_392 = None
        mul_393 = key_states_43 * cos_46
        cos_46 = None
        x1_87 = key_states_43[(Ellipsis, slice(None, 64, None))]
        x2_87 = key_states_43[(Ellipsis, slice(64, None, None))]
        key_states_43 = None
        neg_87 = -x2_87
        x2_87 = None
        cat_88 = torch.cat((neg_87, x1_87), dim=-1)
        neg_87 = x1_87 = None
        mul_394 = cat_88 * sin_46
        cat_88 = sin_46 = None
        k_embed_43 = mul_393 + mul_394
        mul_393 = mul_394 = None
        getitem_307 = k_embed_43[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_43 = None
        hidden_states_433 = getitem_307.expand(1, 8, 5, 19, 128)
        getitem_307 = None
        key_86 = hidden_states_433.reshape(1, 40, 19, 128)
        hidden_states_433 = None
        getitem_308 = value_states_43[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_43 = None
        hidden_states_434 = getitem_308.expand(1, 8, 5, 19, 128)
        getitem_308 = None
        value_86 = hidden_states_434.reshape(1, 40, 19, 128)
        hidden_states_434 = None
        attention_mask_44 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_43 = q_embed_43.contiguous()
        q_embed_43 = None
        key_87 = key_86.contiguous()
        key_86 = None
        value_87 = value_86.contiguous()
        value_86 = None
        attn_output_172 = torch._C._nn.scaled_dot_product_attention(
            query_43,
            key_87,
            value_87,
            attn_mask=attention_mask_44,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_43 = key_87 = value_87 = attention_mask_44 = None
        transpose_176 = attn_output_172.transpose(1, 2)
        attn_output_172 = None
        attn_output_173 = transpose_176.contiguous()
        transpose_176 = None
        reshape_131 = attn_output_173.reshape(1, 19, -1)
        attn_output_173 = None
        attn_output_174 = reshape_131.contiguous()
        reshape_131 = None
        attn_output_175 = torch._C._nn.linear(
            attn_output_174,
            l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_174 = l_self_modules_model_modules_layers_modules_43_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_435 = hidden_states_429 + attn_output_175
        hidden_states_429 = attn_output_175 = None
        hidden_states_436 = hidden_states_435.to(torch.float32)
        pow_88 = hidden_states_436.pow(2)
        variance_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_262 = variance_87 + 1e-05
        variance_87 = None
        rsqrt_87 = torch.rsqrt(add_262)
        add_262 = None
        hidden_states_437 = hidden_states_436 * rsqrt_87
        hidden_states_436 = rsqrt_87 = None
        to_179 = hidden_states_437.to(torch.float16)
        hidden_states_437 = None
        hidden_states_438 = (
            l_self_modules_model_modules_layers_modules_43_modules_post_attention_layernorm_parameters_weight_
            * to_179
        )
        l_self_modules_model_modules_layers_modules_43_modules_post_attention_layernorm_parameters_weight_ = (
            to_179
        ) = None
        linear_305 = torch._C._nn.linear(
            hidden_states_438,
            l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_43 = torch.nn.functional.silu(linear_305, inplace=False)
        linear_305 = None
        linear_306 = torch._C._nn.linear(
            hidden_states_438,
            l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_438 = l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_397 = silu_43 * linear_306
        silu_43 = linear_306 = None
        down_proj_43 = torch._C._nn.linear(
            mul_397,
            l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_397 = l_self_modules_model_modules_layers_modules_43_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_439 = hidden_states_435 + down_proj_43
        hidden_states_435 = down_proj_43 = None
        hidden_states_440 = hidden_states_439.to(torch.float32)
        pow_89 = hidden_states_440.pow(2)
        variance_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_264 = variance_88 + 1e-05
        variance_88 = None
        rsqrt_88 = torch.rsqrt(add_264)
        add_264 = None
        hidden_states_441 = hidden_states_440 * rsqrt_88
        hidden_states_440 = rsqrt_88 = None
        to_181 = hidden_states_441.to(torch.float16)
        hidden_states_441 = None
        hidden_states_442 = (
            l_self_modules_model_modules_layers_modules_44_modules_input_layernorm_parameters_weight_
            * to_181
        )
        l_self_modules_model_modules_layers_modules_44_modules_input_layernorm_parameters_weight_ = (
            to_181
        ) = None
        linear_308 = torch._C._nn.linear(
            hidden_states_442,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_132 = linear_308.view((1, 19, -1, 128))
        linear_308 = None
        query_states_44 = view_132.transpose(1, 2)
        view_132 = None
        linear_309 = torch._C._nn.linear(
            hidden_states_442,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_133 = linear_309.view((1, 19, -1, 128))
        linear_309 = None
        key_states_44 = view_133.transpose(1, 2)
        view_133 = None
        linear_310 = torch._C._nn.linear(
            hidden_states_442,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_442 = l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_134 = linear_310.view((1, 19, -1, 128))
        linear_310 = None
        value_states_44 = view_134.transpose(1, 2)
        view_134 = None
        cos_47 = cos_2.unsqueeze(1)
        sin_47 = sin_2.unsqueeze(1)
        mul_400 = query_states_44 * cos_47
        x1_88 = query_states_44[(Ellipsis, slice(None, 64, None))]
        x2_88 = query_states_44[(Ellipsis, slice(64, None, None))]
        query_states_44 = None
        neg_88 = -x2_88
        x2_88 = None
        cat_89 = torch.cat((neg_88, x1_88), dim=-1)
        neg_88 = x1_88 = None
        mul_401 = cat_89 * sin_47
        cat_89 = None
        q_embed_44 = mul_400 + mul_401
        mul_400 = mul_401 = None
        mul_402 = key_states_44 * cos_47
        cos_47 = None
        x1_89 = key_states_44[(Ellipsis, slice(None, 64, None))]
        x2_89 = key_states_44[(Ellipsis, slice(64, None, None))]
        key_states_44 = None
        neg_89 = -x2_89
        x2_89 = None
        cat_90 = torch.cat((neg_89, x1_89), dim=-1)
        neg_89 = x1_89 = None
        mul_403 = cat_90 * sin_47
        cat_90 = sin_47 = None
        k_embed_44 = mul_402 + mul_403
        mul_402 = mul_403 = None
        getitem_314 = k_embed_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_44 = None
        hidden_states_443 = getitem_314.expand(1, 8, 5, 19, 128)
        getitem_314 = None
        key_88 = hidden_states_443.reshape(1, 40, 19, 128)
        hidden_states_443 = None
        getitem_315 = value_states_44[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_44 = None
        hidden_states_444 = getitem_315.expand(1, 8, 5, 19, 128)
        getitem_315 = None
        value_88 = hidden_states_444.reshape(1, 40, 19, 128)
        hidden_states_444 = None
        attention_mask_45 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_44 = q_embed_44.contiguous()
        q_embed_44 = None
        key_89 = key_88.contiguous()
        key_88 = None
        value_89 = value_88.contiguous()
        value_88 = None
        attn_output_176 = torch._C._nn.scaled_dot_product_attention(
            query_44,
            key_89,
            value_89,
            attn_mask=attention_mask_45,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_44 = key_89 = value_89 = attention_mask_45 = None
        transpose_180 = attn_output_176.transpose(1, 2)
        attn_output_176 = None
        attn_output_177 = transpose_180.contiguous()
        transpose_180 = None
        reshape_134 = attn_output_177.reshape(1, 19, -1)
        attn_output_177 = None
        attn_output_178 = reshape_134.contiguous()
        reshape_134 = None
        attn_output_179 = torch._C._nn.linear(
            attn_output_178,
            l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_178 = l_self_modules_model_modules_layers_modules_44_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_445 = hidden_states_439 + attn_output_179
        hidden_states_439 = attn_output_179 = None
        hidden_states_446 = hidden_states_445.to(torch.float32)
        pow_90 = hidden_states_446.pow(2)
        variance_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_268 = variance_89 + 1e-05
        variance_89 = None
        rsqrt_89 = torch.rsqrt(add_268)
        add_268 = None
        hidden_states_447 = hidden_states_446 * rsqrt_89
        hidden_states_446 = rsqrt_89 = None
        to_183 = hidden_states_447.to(torch.float16)
        hidden_states_447 = None
        hidden_states_448 = (
            l_self_modules_model_modules_layers_modules_44_modules_post_attention_layernorm_parameters_weight_
            * to_183
        )
        l_self_modules_model_modules_layers_modules_44_modules_post_attention_layernorm_parameters_weight_ = (
            to_183
        ) = None
        linear_312 = torch._C._nn.linear(
            hidden_states_448,
            l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_44 = torch.nn.functional.silu(linear_312, inplace=False)
        linear_312 = None
        linear_313 = torch._C._nn.linear(
            hidden_states_448,
            l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_448 = l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_406 = silu_44 * linear_313
        silu_44 = linear_313 = None
        down_proj_44 = torch._C._nn.linear(
            mul_406,
            l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_406 = l_self_modules_model_modules_layers_modules_44_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_449 = hidden_states_445 + down_proj_44
        hidden_states_445 = down_proj_44 = None
        hidden_states_450 = hidden_states_449.to(torch.float32)
        pow_91 = hidden_states_450.pow(2)
        variance_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        add_270 = variance_90 + 1e-05
        variance_90 = None
        rsqrt_90 = torch.rsqrt(add_270)
        add_270 = None
        hidden_states_451 = hidden_states_450 * rsqrt_90
        hidden_states_450 = rsqrt_90 = None
        to_185 = hidden_states_451.to(torch.float16)
        hidden_states_451 = None
        hidden_states_452 = (
            l_self_modules_model_modules_layers_modules_45_modules_input_layernorm_parameters_weight_
            * to_185
        )
        l_self_modules_model_modules_layers_modules_45_modules_input_layernorm_parameters_weight_ = (
            to_185
        ) = None
        linear_315 = torch._C._nn.linear(
            hidden_states_452,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_135 = linear_315.view((1, 19, -1, 128))
        linear_315 = None
        query_states_45 = view_135.transpose(1, 2)
        view_135 = None
        linear_316 = torch._C._nn.linear(
            hidden_states_452,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_136 = linear_316.view((1, 19, -1, 128))
        linear_316 = None
        key_states_45 = view_136.transpose(1, 2)
        view_136 = None
        linear_317 = torch._C._nn.linear(
            hidden_states_452,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_452 = l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_137 = linear_317.view((1, 19, -1, 128))
        linear_317 = None
        value_states_45 = view_137.transpose(1, 2)
        view_137 = None
        cos_48 = cos_2.unsqueeze(1)
        sin_48 = sin_2.unsqueeze(1)
        mul_409 = query_states_45 * cos_48
        x1_90 = query_states_45[(Ellipsis, slice(None, 64, None))]
        x2_90 = query_states_45[(Ellipsis, slice(64, None, None))]
        query_states_45 = None
        neg_90 = -x2_90
        x2_90 = None
        cat_91 = torch.cat((neg_90, x1_90), dim=-1)
        neg_90 = x1_90 = None
        mul_410 = cat_91 * sin_48
        cat_91 = None
        q_embed_45 = mul_409 + mul_410
        mul_409 = mul_410 = None
        mul_411 = key_states_45 * cos_48
        cos_48 = None
        x1_91 = key_states_45[(Ellipsis, slice(None, 64, None))]
        x2_91 = key_states_45[(Ellipsis, slice(64, None, None))]
        key_states_45 = None
        neg_91 = -x2_91
        x2_91 = None
        cat_92 = torch.cat((neg_91, x1_91), dim=-1)
        neg_91 = x1_91 = None
        mul_412 = cat_92 * sin_48
        cat_92 = sin_48 = None
        k_embed_45 = mul_411 + mul_412
        mul_411 = mul_412 = None
        getitem_321 = k_embed_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_45 = None
        hidden_states_453 = getitem_321.expand(1, 8, 5, 19, 128)
        getitem_321 = None
        key_90 = hidden_states_453.reshape(1, 40, 19, 128)
        hidden_states_453 = None
        getitem_322 = value_states_45[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_45 = None
        hidden_states_454 = getitem_322.expand(1, 8, 5, 19, 128)
        getitem_322 = None
        value_90 = hidden_states_454.reshape(1, 40, 19, 128)
        hidden_states_454 = None
        attention_mask_46 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_45 = q_embed_45.contiguous()
        q_embed_45 = None
        key_91 = key_90.contiguous()
        key_90 = None
        value_91 = value_90.contiguous()
        value_90 = None
        attn_output_180 = torch._C._nn.scaled_dot_product_attention(
            query_45,
            key_91,
            value_91,
            attn_mask=attention_mask_46,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_45 = key_91 = value_91 = attention_mask_46 = None
        transpose_184 = attn_output_180.transpose(1, 2)
        attn_output_180 = None
        attn_output_181 = transpose_184.contiguous()
        transpose_184 = None
        reshape_137 = attn_output_181.reshape(1, 19, -1)
        attn_output_181 = None
        attn_output_182 = reshape_137.contiguous()
        reshape_137 = None
        attn_output_183 = torch._C._nn.linear(
            attn_output_182,
            l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_182 = l_self_modules_model_modules_layers_modules_45_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_455 = hidden_states_449 + attn_output_183
        hidden_states_449 = attn_output_183 = None
        hidden_states_456 = hidden_states_455.to(torch.float32)
        pow_92 = hidden_states_456.pow(2)
        variance_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_274 = variance_91 + 1e-05
        variance_91 = None
        rsqrt_91 = torch.rsqrt(add_274)
        add_274 = None
        hidden_states_457 = hidden_states_456 * rsqrt_91
        hidden_states_456 = rsqrt_91 = None
        to_187 = hidden_states_457.to(torch.float16)
        hidden_states_457 = None
        hidden_states_458 = (
            l_self_modules_model_modules_layers_modules_45_modules_post_attention_layernorm_parameters_weight_
            * to_187
        )
        l_self_modules_model_modules_layers_modules_45_modules_post_attention_layernorm_parameters_weight_ = (
            to_187
        ) = None
        linear_319 = torch._C._nn.linear(
            hidden_states_458,
            l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_45 = torch.nn.functional.silu(linear_319, inplace=False)
        linear_319 = None
        linear_320 = torch._C._nn.linear(
            hidden_states_458,
            l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_458 = l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_415 = silu_45 * linear_320
        silu_45 = linear_320 = None
        down_proj_45 = torch._C._nn.linear(
            mul_415,
            l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_415 = l_self_modules_model_modules_layers_modules_45_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_459 = hidden_states_455 + down_proj_45
        hidden_states_455 = down_proj_45 = None
        hidden_states_460 = hidden_states_459.to(torch.float32)
        pow_93 = hidden_states_460.pow(2)
        variance_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_276 = variance_92 + 1e-05
        variance_92 = None
        rsqrt_92 = torch.rsqrt(add_276)
        add_276 = None
        hidden_states_461 = hidden_states_460 * rsqrt_92
        hidden_states_460 = rsqrt_92 = None
        to_189 = hidden_states_461.to(torch.float16)
        hidden_states_461 = None
        hidden_states_462 = (
            l_self_modules_model_modules_layers_modules_46_modules_input_layernorm_parameters_weight_
            * to_189
        )
        l_self_modules_model_modules_layers_modules_46_modules_input_layernorm_parameters_weight_ = (
            to_189
        ) = None
        linear_322 = torch._C._nn.linear(
            hidden_states_462,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_138 = linear_322.view((1, 19, -1, 128))
        linear_322 = None
        query_states_46 = view_138.transpose(1, 2)
        view_138 = None
        linear_323 = torch._C._nn.linear(
            hidden_states_462,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_139 = linear_323.view((1, 19, -1, 128))
        linear_323 = None
        key_states_46 = view_139.transpose(1, 2)
        view_139 = None
        linear_324 = torch._C._nn.linear(
            hidden_states_462,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_462 = l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_140 = linear_324.view((1, 19, -1, 128))
        linear_324 = None
        value_states_46 = view_140.transpose(1, 2)
        view_140 = None
        cos_49 = cos_2.unsqueeze(1)
        sin_49 = sin_2.unsqueeze(1)
        mul_418 = query_states_46 * cos_49
        x1_92 = query_states_46[(Ellipsis, slice(None, 64, None))]
        x2_92 = query_states_46[(Ellipsis, slice(64, None, None))]
        query_states_46 = None
        neg_92 = -x2_92
        x2_92 = None
        cat_93 = torch.cat((neg_92, x1_92), dim=-1)
        neg_92 = x1_92 = None
        mul_419 = cat_93 * sin_49
        cat_93 = None
        q_embed_46 = mul_418 + mul_419
        mul_418 = mul_419 = None
        mul_420 = key_states_46 * cos_49
        cos_49 = None
        x1_93 = key_states_46[(Ellipsis, slice(None, 64, None))]
        x2_93 = key_states_46[(Ellipsis, slice(64, None, None))]
        key_states_46 = None
        neg_93 = -x2_93
        x2_93 = None
        cat_94 = torch.cat((neg_93, x1_93), dim=-1)
        neg_93 = x1_93 = None
        mul_421 = cat_94 * sin_49
        cat_94 = sin_49 = None
        k_embed_46 = mul_420 + mul_421
        mul_420 = mul_421 = None
        getitem_328 = k_embed_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_46 = None
        hidden_states_463 = getitem_328.expand(1, 8, 5, 19, 128)
        getitem_328 = None
        key_92 = hidden_states_463.reshape(1, 40, 19, 128)
        hidden_states_463 = None
        getitem_329 = value_states_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_46 = None
        hidden_states_464 = getitem_329.expand(1, 8, 5, 19, 128)
        getitem_329 = None
        value_92 = hidden_states_464.reshape(1, 40, 19, 128)
        hidden_states_464 = None
        attention_mask_47 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_46 = q_embed_46.contiguous()
        q_embed_46 = None
        key_93 = key_92.contiguous()
        key_92 = None
        value_93 = value_92.contiguous()
        value_92 = None
        attn_output_184 = torch._C._nn.scaled_dot_product_attention(
            query_46,
            key_93,
            value_93,
            attn_mask=attention_mask_47,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_46 = key_93 = value_93 = attention_mask_47 = None
        transpose_188 = attn_output_184.transpose(1, 2)
        attn_output_184 = None
        attn_output_185 = transpose_188.contiguous()
        transpose_188 = None
        reshape_140 = attn_output_185.reshape(1, 19, -1)
        attn_output_185 = None
        attn_output_186 = reshape_140.contiguous()
        reshape_140 = None
        attn_output_187 = torch._C._nn.linear(
            attn_output_186,
            l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_186 = l_self_modules_model_modules_layers_modules_46_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_465 = hidden_states_459 + attn_output_187
        hidden_states_459 = attn_output_187 = None
        hidden_states_466 = hidden_states_465.to(torch.float32)
        pow_94 = hidden_states_466.pow(2)
        variance_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_280 = variance_93 + 1e-05
        variance_93 = None
        rsqrt_93 = torch.rsqrt(add_280)
        add_280 = None
        hidden_states_467 = hidden_states_466 * rsqrt_93
        hidden_states_466 = rsqrt_93 = None
        to_191 = hidden_states_467.to(torch.float16)
        hidden_states_467 = None
        hidden_states_468 = (
            l_self_modules_model_modules_layers_modules_46_modules_post_attention_layernorm_parameters_weight_
            * to_191
        )
        l_self_modules_model_modules_layers_modules_46_modules_post_attention_layernorm_parameters_weight_ = (
            to_191
        ) = None
        linear_326 = torch._C._nn.linear(
            hidden_states_468,
            l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_46 = torch.nn.functional.silu(linear_326, inplace=False)
        linear_326 = None
        linear_327 = torch._C._nn.linear(
            hidden_states_468,
            l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_468 = l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_424 = silu_46 * linear_327
        silu_46 = linear_327 = None
        down_proj_46 = torch._C._nn.linear(
            mul_424,
            l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_424 = l_self_modules_model_modules_layers_modules_46_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_469 = hidden_states_465 + down_proj_46
        hidden_states_465 = down_proj_46 = None
        hidden_states_470 = hidden_states_469.to(torch.float32)
        pow_95 = hidden_states_470.pow(2)
        variance_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        add_282 = variance_94 + 1e-05
        variance_94 = None
        rsqrt_94 = torch.rsqrt(add_282)
        add_282 = None
        hidden_states_471 = hidden_states_470 * rsqrt_94
        hidden_states_470 = rsqrt_94 = None
        to_193 = hidden_states_471.to(torch.float16)
        hidden_states_471 = None
        hidden_states_472 = (
            l_self_modules_model_modules_layers_modules_47_modules_input_layernorm_parameters_weight_
            * to_193
        )
        l_self_modules_model_modules_layers_modules_47_modules_input_layernorm_parameters_weight_ = (
            to_193
        ) = None
        linear_329 = torch._C._nn.linear(
            hidden_states_472,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_141 = linear_329.view((1, 19, -1, 128))
        linear_329 = None
        query_states_47 = view_141.transpose(1, 2)
        view_141 = None
        linear_330 = torch._C._nn.linear(
            hidden_states_472,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_142 = linear_330.view((1, 19, -1, 128))
        linear_330 = None
        key_states_47 = view_142.transpose(1, 2)
        view_142 = None
        linear_331 = torch._C._nn.linear(
            hidden_states_472,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_472 = l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_143 = linear_331.view((1, 19, -1, 128))
        linear_331 = None
        value_states_47 = view_143.transpose(1, 2)
        view_143 = None
        cos_50 = cos_2.unsqueeze(1)
        sin_50 = sin_2.unsqueeze(1)
        mul_427 = query_states_47 * cos_50
        x1_94 = query_states_47[(Ellipsis, slice(None, 64, None))]
        x2_94 = query_states_47[(Ellipsis, slice(64, None, None))]
        query_states_47 = None
        neg_94 = -x2_94
        x2_94 = None
        cat_95 = torch.cat((neg_94, x1_94), dim=-1)
        neg_94 = x1_94 = None
        mul_428 = cat_95 * sin_50
        cat_95 = None
        q_embed_47 = mul_427 + mul_428
        mul_427 = mul_428 = None
        mul_429 = key_states_47 * cos_50
        cos_50 = None
        x1_95 = key_states_47[(Ellipsis, slice(None, 64, None))]
        x2_95 = key_states_47[(Ellipsis, slice(64, None, None))]
        key_states_47 = None
        neg_95 = -x2_95
        x2_95 = None
        cat_96 = torch.cat((neg_95, x1_95), dim=-1)
        neg_95 = x1_95 = None
        mul_430 = cat_96 * sin_50
        cat_96 = sin_50 = None
        k_embed_47 = mul_429 + mul_430
        mul_429 = mul_430 = None
        getitem_335 = k_embed_47[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_47 = None
        hidden_states_473 = getitem_335.expand(1, 8, 5, 19, 128)
        getitem_335 = None
        key_94 = hidden_states_473.reshape(1, 40, 19, 128)
        hidden_states_473 = None
        getitem_336 = value_states_47[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_47 = None
        hidden_states_474 = getitem_336.expand(1, 8, 5, 19, 128)
        getitem_336 = None
        value_94 = hidden_states_474.reshape(1, 40, 19, 128)
        hidden_states_474 = None
        attention_mask_48 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_47 = q_embed_47.contiguous()
        q_embed_47 = None
        key_95 = key_94.contiguous()
        key_94 = None
        value_95 = value_94.contiguous()
        value_94 = None
        attn_output_188 = torch._C._nn.scaled_dot_product_attention(
            query_47,
            key_95,
            value_95,
            attn_mask=attention_mask_48,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_47 = key_95 = value_95 = attention_mask_48 = None
        transpose_192 = attn_output_188.transpose(1, 2)
        attn_output_188 = None
        attn_output_189 = transpose_192.contiguous()
        transpose_192 = None
        reshape_143 = attn_output_189.reshape(1, 19, -1)
        attn_output_189 = None
        attn_output_190 = reshape_143.contiguous()
        reshape_143 = None
        attn_output_191 = torch._C._nn.linear(
            attn_output_190,
            l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_190 = l_self_modules_model_modules_layers_modules_47_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_475 = hidden_states_469 + attn_output_191
        hidden_states_469 = attn_output_191 = None
        hidden_states_476 = hidden_states_475.to(torch.float32)
        pow_96 = hidden_states_476.pow(2)
        variance_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_286 = variance_95 + 1e-05
        variance_95 = None
        rsqrt_95 = torch.rsqrt(add_286)
        add_286 = None
        hidden_states_477 = hidden_states_476 * rsqrt_95
        hidden_states_476 = rsqrt_95 = None
        to_195 = hidden_states_477.to(torch.float16)
        hidden_states_477 = None
        hidden_states_478 = (
            l_self_modules_model_modules_layers_modules_47_modules_post_attention_layernorm_parameters_weight_
            * to_195
        )
        l_self_modules_model_modules_layers_modules_47_modules_post_attention_layernorm_parameters_weight_ = (
            to_195
        ) = None
        linear_333 = torch._C._nn.linear(
            hidden_states_478,
            l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_47 = torch.nn.functional.silu(linear_333, inplace=False)
        linear_333 = None
        linear_334 = torch._C._nn.linear(
            hidden_states_478,
            l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_478 = l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_433 = silu_47 * linear_334
        silu_47 = linear_334 = None
        down_proj_47 = torch._C._nn.linear(
            mul_433,
            l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_433 = l_self_modules_model_modules_layers_modules_47_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_479 = hidden_states_475 + down_proj_47
        hidden_states_475 = down_proj_47 = None
        hidden_states_480 = hidden_states_479.to(torch.float32)
        pow_97 = hidden_states_480.pow(2)
        variance_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_288 = variance_96 + 1e-05
        variance_96 = None
        rsqrt_96 = torch.rsqrt(add_288)
        add_288 = None
        hidden_states_481 = hidden_states_480 * rsqrt_96
        hidden_states_480 = rsqrt_96 = None
        to_197 = hidden_states_481.to(torch.float16)
        hidden_states_481 = None
        hidden_states_482 = (
            l_self_modules_model_modules_layers_modules_48_modules_input_layernorm_parameters_weight_
            * to_197
        )
        l_self_modules_model_modules_layers_modules_48_modules_input_layernorm_parameters_weight_ = (
            to_197
        ) = None
        linear_336 = torch._C._nn.linear(
            hidden_states_482,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_144 = linear_336.view((1, 19, -1, 128))
        linear_336 = None
        query_states_48 = view_144.transpose(1, 2)
        view_144 = None
        linear_337 = torch._C._nn.linear(
            hidden_states_482,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_145 = linear_337.view((1, 19, -1, 128))
        linear_337 = None
        key_states_48 = view_145.transpose(1, 2)
        view_145 = None
        linear_338 = torch._C._nn.linear(
            hidden_states_482,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_482 = l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_146 = linear_338.view((1, 19, -1, 128))
        linear_338 = None
        value_states_48 = view_146.transpose(1, 2)
        view_146 = None
        cos_51 = cos_2.unsqueeze(1)
        sin_51 = sin_2.unsqueeze(1)
        mul_436 = query_states_48 * cos_51
        x1_96 = query_states_48[(Ellipsis, slice(None, 64, None))]
        x2_96 = query_states_48[(Ellipsis, slice(64, None, None))]
        query_states_48 = None
        neg_96 = -x2_96
        x2_96 = None
        cat_97 = torch.cat((neg_96, x1_96), dim=-1)
        neg_96 = x1_96 = None
        mul_437 = cat_97 * sin_51
        cat_97 = None
        q_embed_48 = mul_436 + mul_437
        mul_436 = mul_437 = None
        mul_438 = key_states_48 * cos_51
        cos_51 = None
        x1_97 = key_states_48[(Ellipsis, slice(None, 64, None))]
        x2_97 = key_states_48[(Ellipsis, slice(64, None, None))]
        key_states_48 = None
        neg_97 = -x2_97
        x2_97 = None
        cat_98 = torch.cat((neg_97, x1_97), dim=-1)
        neg_97 = x1_97 = None
        mul_439 = cat_98 * sin_51
        cat_98 = sin_51 = None
        k_embed_48 = mul_438 + mul_439
        mul_438 = mul_439 = None
        getitem_342 = k_embed_48[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_48 = None
        hidden_states_483 = getitem_342.expand(1, 8, 5, 19, 128)
        getitem_342 = None
        key_96 = hidden_states_483.reshape(1, 40, 19, 128)
        hidden_states_483 = None
        getitem_343 = value_states_48[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_48 = None
        hidden_states_484 = getitem_343.expand(1, 8, 5, 19, 128)
        getitem_343 = None
        value_96 = hidden_states_484.reshape(1, 40, 19, 128)
        hidden_states_484 = None
        attention_mask_49 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_48 = q_embed_48.contiguous()
        q_embed_48 = None
        key_97 = key_96.contiguous()
        key_96 = None
        value_97 = value_96.contiguous()
        value_96 = None
        attn_output_192 = torch._C._nn.scaled_dot_product_attention(
            query_48,
            key_97,
            value_97,
            attn_mask=attention_mask_49,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_48 = key_97 = value_97 = attention_mask_49 = None
        transpose_196 = attn_output_192.transpose(1, 2)
        attn_output_192 = None
        attn_output_193 = transpose_196.contiguous()
        transpose_196 = None
        reshape_146 = attn_output_193.reshape(1, 19, -1)
        attn_output_193 = None
        attn_output_194 = reshape_146.contiguous()
        reshape_146 = None
        attn_output_195 = torch._C._nn.linear(
            attn_output_194,
            l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_194 = l_self_modules_model_modules_layers_modules_48_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_485 = hidden_states_479 + attn_output_195
        hidden_states_479 = attn_output_195 = None
        hidden_states_486 = hidden_states_485.to(torch.float32)
        pow_98 = hidden_states_486.pow(2)
        variance_97 = pow_98.mean(-1, keepdim=True)
        pow_98 = None
        add_292 = variance_97 + 1e-05
        variance_97 = None
        rsqrt_97 = torch.rsqrt(add_292)
        add_292 = None
        hidden_states_487 = hidden_states_486 * rsqrt_97
        hidden_states_486 = rsqrt_97 = None
        to_199 = hidden_states_487.to(torch.float16)
        hidden_states_487 = None
        hidden_states_488 = (
            l_self_modules_model_modules_layers_modules_48_modules_post_attention_layernorm_parameters_weight_
            * to_199
        )
        l_self_modules_model_modules_layers_modules_48_modules_post_attention_layernorm_parameters_weight_ = (
            to_199
        ) = None
        linear_340 = torch._C._nn.linear(
            hidden_states_488,
            l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_48 = torch.nn.functional.silu(linear_340, inplace=False)
        linear_340 = None
        linear_341 = torch._C._nn.linear(
            hidden_states_488,
            l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_488 = l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_442 = silu_48 * linear_341
        silu_48 = linear_341 = None
        down_proj_48 = torch._C._nn.linear(
            mul_442,
            l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_442 = l_self_modules_model_modules_layers_modules_48_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_489 = hidden_states_485 + down_proj_48
        hidden_states_485 = down_proj_48 = None
        hidden_states_490 = hidden_states_489.to(torch.float32)
        pow_99 = hidden_states_490.pow(2)
        variance_98 = pow_99.mean(-1, keepdim=True)
        pow_99 = None
        add_294 = variance_98 + 1e-05
        variance_98 = None
        rsqrt_98 = torch.rsqrt(add_294)
        add_294 = None
        hidden_states_491 = hidden_states_490 * rsqrt_98
        hidden_states_490 = rsqrt_98 = None
        to_201 = hidden_states_491.to(torch.float16)
        hidden_states_491 = None
        hidden_states_492 = (
            l_self_modules_model_modules_layers_modules_49_modules_input_layernorm_parameters_weight_
            * to_201
        )
        l_self_modules_model_modules_layers_modules_49_modules_input_layernorm_parameters_weight_ = (
            to_201
        ) = None
        linear_343 = torch._C._nn.linear(
            hidden_states_492,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_147 = linear_343.view((1, 19, -1, 128))
        linear_343 = None
        query_states_49 = view_147.transpose(1, 2)
        view_147 = None
        linear_344 = torch._C._nn.linear(
            hidden_states_492,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_148 = linear_344.view((1, 19, -1, 128))
        linear_344 = None
        key_states_49 = view_148.transpose(1, 2)
        view_148 = None
        linear_345 = torch._C._nn.linear(
            hidden_states_492,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_492 = l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_149 = linear_345.view((1, 19, -1, 128))
        linear_345 = None
        value_states_49 = view_149.transpose(1, 2)
        view_149 = None
        cos_52 = cos_2.unsqueeze(1)
        sin_52 = sin_2.unsqueeze(1)
        mul_445 = query_states_49 * cos_52
        x1_98 = query_states_49[(Ellipsis, slice(None, 64, None))]
        x2_98 = query_states_49[(Ellipsis, slice(64, None, None))]
        query_states_49 = None
        neg_98 = -x2_98
        x2_98 = None
        cat_99 = torch.cat((neg_98, x1_98), dim=-1)
        neg_98 = x1_98 = None
        mul_446 = cat_99 * sin_52
        cat_99 = None
        q_embed_49 = mul_445 + mul_446
        mul_445 = mul_446 = None
        mul_447 = key_states_49 * cos_52
        cos_52 = None
        x1_99 = key_states_49[(Ellipsis, slice(None, 64, None))]
        x2_99 = key_states_49[(Ellipsis, slice(64, None, None))]
        key_states_49 = None
        neg_99 = -x2_99
        x2_99 = None
        cat_100 = torch.cat((neg_99, x1_99), dim=-1)
        neg_99 = x1_99 = None
        mul_448 = cat_100 * sin_52
        cat_100 = sin_52 = None
        k_embed_49 = mul_447 + mul_448
        mul_447 = mul_448 = None
        getitem_349 = k_embed_49[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_49 = None
        hidden_states_493 = getitem_349.expand(1, 8, 5, 19, 128)
        getitem_349 = None
        key_98 = hidden_states_493.reshape(1, 40, 19, 128)
        hidden_states_493 = None
        getitem_350 = value_states_49[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_49 = None
        hidden_states_494 = getitem_350.expand(1, 8, 5, 19, 128)
        getitem_350 = None
        value_98 = hidden_states_494.reshape(1, 40, 19, 128)
        hidden_states_494 = None
        attention_mask_50 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_49 = q_embed_49.contiguous()
        q_embed_49 = None
        key_99 = key_98.contiguous()
        key_98 = None
        value_99 = value_98.contiguous()
        value_98 = None
        attn_output_196 = torch._C._nn.scaled_dot_product_attention(
            query_49,
            key_99,
            value_99,
            attn_mask=attention_mask_50,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_49 = key_99 = value_99 = attention_mask_50 = None
        transpose_200 = attn_output_196.transpose(1, 2)
        attn_output_196 = None
        attn_output_197 = transpose_200.contiguous()
        transpose_200 = None
        reshape_149 = attn_output_197.reshape(1, 19, -1)
        attn_output_197 = None
        attn_output_198 = reshape_149.contiguous()
        reshape_149 = None
        attn_output_199 = torch._C._nn.linear(
            attn_output_198,
            l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_198 = l_self_modules_model_modules_layers_modules_49_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_495 = hidden_states_489 + attn_output_199
        hidden_states_489 = attn_output_199 = None
        hidden_states_496 = hidden_states_495.to(torch.float32)
        pow_100 = hidden_states_496.pow(2)
        variance_99 = pow_100.mean(-1, keepdim=True)
        pow_100 = None
        add_298 = variance_99 + 1e-05
        variance_99 = None
        rsqrt_99 = torch.rsqrt(add_298)
        add_298 = None
        hidden_states_497 = hidden_states_496 * rsqrt_99
        hidden_states_496 = rsqrt_99 = None
        to_203 = hidden_states_497.to(torch.float16)
        hidden_states_497 = None
        hidden_states_498 = (
            l_self_modules_model_modules_layers_modules_49_modules_post_attention_layernorm_parameters_weight_
            * to_203
        )
        l_self_modules_model_modules_layers_modules_49_modules_post_attention_layernorm_parameters_weight_ = (
            to_203
        ) = None
        linear_347 = torch._C._nn.linear(
            hidden_states_498,
            l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_49 = torch.nn.functional.silu(linear_347, inplace=False)
        linear_347 = None
        linear_348 = torch._C._nn.linear(
            hidden_states_498,
            l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_498 = l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_451 = silu_49 * linear_348
        silu_49 = linear_348 = None
        down_proj_49 = torch._C._nn.linear(
            mul_451,
            l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_451 = l_self_modules_model_modules_layers_modules_49_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_499 = hidden_states_495 + down_proj_49
        hidden_states_495 = down_proj_49 = None
        hidden_states_500 = hidden_states_499.to(torch.float32)
        pow_101 = hidden_states_500.pow(2)
        variance_100 = pow_101.mean(-1, keepdim=True)
        pow_101 = None
        add_300 = variance_100 + 1e-05
        variance_100 = None
        rsqrt_100 = torch.rsqrt(add_300)
        add_300 = None
        hidden_states_501 = hidden_states_500 * rsqrt_100
        hidden_states_500 = rsqrt_100 = None
        to_205 = hidden_states_501.to(torch.float16)
        hidden_states_501 = None
        hidden_states_502 = (
            l_self_modules_model_modules_layers_modules_50_modules_input_layernorm_parameters_weight_
            * to_205
        )
        l_self_modules_model_modules_layers_modules_50_modules_input_layernorm_parameters_weight_ = (
            to_205
        ) = None
        linear_350 = torch._C._nn.linear(
            hidden_states_502,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_150 = linear_350.view((1, 19, -1, 128))
        linear_350 = None
        query_states_50 = view_150.transpose(1, 2)
        view_150 = None
        linear_351 = torch._C._nn.linear(
            hidden_states_502,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_151 = linear_351.view((1, 19, -1, 128))
        linear_351 = None
        key_states_50 = view_151.transpose(1, 2)
        view_151 = None
        linear_352 = torch._C._nn.linear(
            hidden_states_502,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_502 = l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_152 = linear_352.view((1, 19, -1, 128))
        linear_352 = None
        value_states_50 = view_152.transpose(1, 2)
        view_152 = None
        cos_53 = cos_2.unsqueeze(1)
        sin_53 = sin_2.unsqueeze(1)
        mul_454 = query_states_50 * cos_53
        x1_100 = query_states_50[(Ellipsis, slice(None, 64, None))]
        x2_100 = query_states_50[(Ellipsis, slice(64, None, None))]
        query_states_50 = None
        neg_100 = -x2_100
        x2_100 = None
        cat_101 = torch.cat((neg_100, x1_100), dim=-1)
        neg_100 = x1_100 = None
        mul_455 = cat_101 * sin_53
        cat_101 = None
        q_embed_50 = mul_454 + mul_455
        mul_454 = mul_455 = None
        mul_456 = key_states_50 * cos_53
        cos_53 = None
        x1_101 = key_states_50[(Ellipsis, slice(None, 64, None))]
        x2_101 = key_states_50[(Ellipsis, slice(64, None, None))]
        key_states_50 = None
        neg_101 = -x2_101
        x2_101 = None
        cat_102 = torch.cat((neg_101, x1_101), dim=-1)
        neg_101 = x1_101 = None
        mul_457 = cat_102 * sin_53
        cat_102 = sin_53 = None
        k_embed_50 = mul_456 + mul_457
        mul_456 = mul_457 = None
        getitem_356 = k_embed_50[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_50 = None
        hidden_states_503 = getitem_356.expand(1, 8, 5, 19, 128)
        getitem_356 = None
        key_100 = hidden_states_503.reshape(1, 40, 19, 128)
        hidden_states_503 = None
        getitem_357 = value_states_50[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_50 = None
        hidden_states_504 = getitem_357.expand(1, 8, 5, 19, 128)
        getitem_357 = None
        value_100 = hidden_states_504.reshape(1, 40, 19, 128)
        hidden_states_504 = None
        attention_mask_51 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_50 = q_embed_50.contiguous()
        q_embed_50 = None
        key_101 = key_100.contiguous()
        key_100 = None
        value_101 = value_100.contiguous()
        value_100 = None
        attn_output_200 = torch._C._nn.scaled_dot_product_attention(
            query_50,
            key_101,
            value_101,
            attn_mask=attention_mask_51,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_50 = key_101 = value_101 = attention_mask_51 = None
        transpose_204 = attn_output_200.transpose(1, 2)
        attn_output_200 = None
        attn_output_201 = transpose_204.contiguous()
        transpose_204 = None
        reshape_152 = attn_output_201.reshape(1, 19, -1)
        attn_output_201 = None
        attn_output_202 = reshape_152.contiguous()
        reshape_152 = None
        attn_output_203 = torch._C._nn.linear(
            attn_output_202,
            l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_202 = l_self_modules_model_modules_layers_modules_50_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_505 = hidden_states_499 + attn_output_203
        hidden_states_499 = attn_output_203 = None
        hidden_states_506 = hidden_states_505.to(torch.float32)
        pow_102 = hidden_states_506.pow(2)
        variance_101 = pow_102.mean(-1, keepdim=True)
        pow_102 = None
        add_304 = variance_101 + 1e-05
        variance_101 = None
        rsqrt_101 = torch.rsqrt(add_304)
        add_304 = None
        hidden_states_507 = hidden_states_506 * rsqrt_101
        hidden_states_506 = rsqrt_101 = None
        to_207 = hidden_states_507.to(torch.float16)
        hidden_states_507 = None
        hidden_states_508 = (
            l_self_modules_model_modules_layers_modules_50_modules_post_attention_layernorm_parameters_weight_
            * to_207
        )
        l_self_modules_model_modules_layers_modules_50_modules_post_attention_layernorm_parameters_weight_ = (
            to_207
        ) = None
        linear_354 = torch._C._nn.linear(
            hidden_states_508,
            l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_50 = torch.nn.functional.silu(linear_354, inplace=False)
        linear_354 = None
        linear_355 = torch._C._nn.linear(
            hidden_states_508,
            l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_508 = l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_460 = silu_50 * linear_355
        silu_50 = linear_355 = None
        down_proj_50 = torch._C._nn.linear(
            mul_460,
            l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_460 = l_self_modules_model_modules_layers_modules_50_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_509 = hidden_states_505 + down_proj_50
        hidden_states_505 = down_proj_50 = None
        hidden_states_510 = hidden_states_509.to(torch.float32)
        pow_103 = hidden_states_510.pow(2)
        variance_102 = pow_103.mean(-1, keepdim=True)
        pow_103 = None
        add_306 = variance_102 + 1e-05
        variance_102 = None
        rsqrt_102 = torch.rsqrt(add_306)
        add_306 = None
        hidden_states_511 = hidden_states_510 * rsqrt_102
        hidden_states_510 = rsqrt_102 = None
        to_209 = hidden_states_511.to(torch.float16)
        hidden_states_511 = None
        hidden_states_512 = (
            l_self_modules_model_modules_layers_modules_51_modules_input_layernorm_parameters_weight_
            * to_209
        )
        l_self_modules_model_modules_layers_modules_51_modules_input_layernorm_parameters_weight_ = (
            to_209
        ) = None
        linear_357 = torch._C._nn.linear(
            hidden_states_512,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_153 = linear_357.view((1, 19, -1, 128))
        linear_357 = None
        query_states_51 = view_153.transpose(1, 2)
        view_153 = None
        linear_358 = torch._C._nn.linear(
            hidden_states_512,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_154 = linear_358.view((1, 19, -1, 128))
        linear_358 = None
        key_states_51 = view_154.transpose(1, 2)
        view_154 = None
        linear_359 = torch._C._nn.linear(
            hidden_states_512,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_512 = l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_155 = linear_359.view((1, 19, -1, 128))
        linear_359 = None
        value_states_51 = view_155.transpose(1, 2)
        view_155 = None
        cos_54 = cos_2.unsqueeze(1)
        sin_54 = sin_2.unsqueeze(1)
        mul_463 = query_states_51 * cos_54
        x1_102 = query_states_51[(Ellipsis, slice(None, 64, None))]
        x2_102 = query_states_51[(Ellipsis, slice(64, None, None))]
        query_states_51 = None
        neg_102 = -x2_102
        x2_102 = None
        cat_103 = torch.cat((neg_102, x1_102), dim=-1)
        neg_102 = x1_102 = None
        mul_464 = cat_103 * sin_54
        cat_103 = None
        q_embed_51 = mul_463 + mul_464
        mul_463 = mul_464 = None
        mul_465 = key_states_51 * cos_54
        cos_54 = None
        x1_103 = key_states_51[(Ellipsis, slice(None, 64, None))]
        x2_103 = key_states_51[(Ellipsis, slice(64, None, None))]
        key_states_51 = None
        neg_103 = -x2_103
        x2_103 = None
        cat_104 = torch.cat((neg_103, x1_103), dim=-1)
        neg_103 = x1_103 = None
        mul_466 = cat_104 * sin_54
        cat_104 = sin_54 = None
        k_embed_51 = mul_465 + mul_466
        mul_465 = mul_466 = None
        getitem_363 = k_embed_51[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_51 = None
        hidden_states_513 = getitem_363.expand(1, 8, 5, 19, 128)
        getitem_363 = None
        key_102 = hidden_states_513.reshape(1, 40, 19, 128)
        hidden_states_513 = None
        getitem_364 = value_states_51[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_51 = None
        hidden_states_514 = getitem_364.expand(1, 8, 5, 19, 128)
        getitem_364 = None
        value_102 = hidden_states_514.reshape(1, 40, 19, 128)
        hidden_states_514 = None
        attention_mask_52 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_51 = q_embed_51.contiguous()
        q_embed_51 = None
        key_103 = key_102.contiguous()
        key_102 = None
        value_103 = value_102.contiguous()
        value_102 = None
        attn_output_204 = torch._C._nn.scaled_dot_product_attention(
            query_51,
            key_103,
            value_103,
            attn_mask=attention_mask_52,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_51 = key_103 = value_103 = attention_mask_52 = None
        transpose_208 = attn_output_204.transpose(1, 2)
        attn_output_204 = None
        attn_output_205 = transpose_208.contiguous()
        transpose_208 = None
        reshape_155 = attn_output_205.reshape(1, 19, -1)
        attn_output_205 = None
        attn_output_206 = reshape_155.contiguous()
        reshape_155 = None
        attn_output_207 = torch._C._nn.linear(
            attn_output_206,
            l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_206 = l_self_modules_model_modules_layers_modules_51_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_515 = hidden_states_509 + attn_output_207
        hidden_states_509 = attn_output_207 = None
        hidden_states_516 = hidden_states_515.to(torch.float32)
        pow_104 = hidden_states_516.pow(2)
        variance_103 = pow_104.mean(-1, keepdim=True)
        pow_104 = None
        add_310 = variance_103 + 1e-05
        variance_103 = None
        rsqrt_103 = torch.rsqrt(add_310)
        add_310 = None
        hidden_states_517 = hidden_states_516 * rsqrt_103
        hidden_states_516 = rsqrt_103 = None
        to_211 = hidden_states_517.to(torch.float16)
        hidden_states_517 = None
        hidden_states_518 = (
            l_self_modules_model_modules_layers_modules_51_modules_post_attention_layernorm_parameters_weight_
            * to_211
        )
        l_self_modules_model_modules_layers_modules_51_modules_post_attention_layernorm_parameters_weight_ = (
            to_211
        ) = None
        linear_361 = torch._C._nn.linear(
            hidden_states_518,
            l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_51 = torch.nn.functional.silu(linear_361, inplace=False)
        linear_361 = None
        linear_362 = torch._C._nn.linear(
            hidden_states_518,
            l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_518 = l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_469 = silu_51 * linear_362
        silu_51 = linear_362 = None
        down_proj_51 = torch._C._nn.linear(
            mul_469,
            l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_469 = l_self_modules_model_modules_layers_modules_51_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_519 = hidden_states_515 + down_proj_51
        hidden_states_515 = down_proj_51 = None
        hidden_states_520 = hidden_states_519.to(torch.float32)
        pow_105 = hidden_states_520.pow(2)
        variance_104 = pow_105.mean(-1, keepdim=True)
        pow_105 = None
        add_312 = variance_104 + 1e-05
        variance_104 = None
        rsqrt_104 = torch.rsqrt(add_312)
        add_312 = None
        hidden_states_521 = hidden_states_520 * rsqrt_104
        hidden_states_520 = rsqrt_104 = None
        to_213 = hidden_states_521.to(torch.float16)
        hidden_states_521 = None
        hidden_states_522 = (
            l_self_modules_model_modules_layers_modules_52_modules_input_layernorm_parameters_weight_
            * to_213
        )
        l_self_modules_model_modules_layers_modules_52_modules_input_layernorm_parameters_weight_ = (
            to_213
        ) = None
        linear_364 = torch._C._nn.linear(
            hidden_states_522,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_156 = linear_364.view((1, 19, -1, 128))
        linear_364 = None
        query_states_52 = view_156.transpose(1, 2)
        view_156 = None
        linear_365 = torch._C._nn.linear(
            hidden_states_522,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_157 = linear_365.view((1, 19, -1, 128))
        linear_365 = None
        key_states_52 = view_157.transpose(1, 2)
        view_157 = None
        linear_366 = torch._C._nn.linear(
            hidden_states_522,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_522 = l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_158 = linear_366.view((1, 19, -1, 128))
        linear_366 = None
        value_states_52 = view_158.transpose(1, 2)
        view_158 = None
        cos_55 = cos_2.unsqueeze(1)
        sin_55 = sin_2.unsqueeze(1)
        mul_472 = query_states_52 * cos_55
        x1_104 = query_states_52[(Ellipsis, slice(None, 64, None))]
        x2_104 = query_states_52[(Ellipsis, slice(64, None, None))]
        query_states_52 = None
        neg_104 = -x2_104
        x2_104 = None
        cat_105 = torch.cat((neg_104, x1_104), dim=-1)
        neg_104 = x1_104 = None
        mul_473 = cat_105 * sin_55
        cat_105 = None
        q_embed_52 = mul_472 + mul_473
        mul_472 = mul_473 = None
        mul_474 = key_states_52 * cos_55
        cos_55 = None
        x1_105 = key_states_52[(Ellipsis, slice(None, 64, None))]
        x2_105 = key_states_52[(Ellipsis, slice(64, None, None))]
        key_states_52 = None
        neg_105 = -x2_105
        x2_105 = None
        cat_106 = torch.cat((neg_105, x1_105), dim=-1)
        neg_105 = x1_105 = None
        mul_475 = cat_106 * sin_55
        cat_106 = sin_55 = None
        k_embed_52 = mul_474 + mul_475
        mul_474 = mul_475 = None
        getitem_370 = k_embed_52[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_52 = None
        hidden_states_523 = getitem_370.expand(1, 8, 5, 19, 128)
        getitem_370 = None
        key_104 = hidden_states_523.reshape(1, 40, 19, 128)
        hidden_states_523 = None
        getitem_371 = value_states_52[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_52 = None
        hidden_states_524 = getitem_371.expand(1, 8, 5, 19, 128)
        getitem_371 = None
        value_104 = hidden_states_524.reshape(1, 40, 19, 128)
        hidden_states_524 = None
        attention_mask_53 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_52 = q_embed_52.contiguous()
        q_embed_52 = None
        key_105 = key_104.contiguous()
        key_104 = None
        value_105 = value_104.contiguous()
        value_104 = None
        attn_output_208 = torch._C._nn.scaled_dot_product_attention(
            query_52,
            key_105,
            value_105,
            attn_mask=attention_mask_53,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_52 = key_105 = value_105 = attention_mask_53 = None
        transpose_212 = attn_output_208.transpose(1, 2)
        attn_output_208 = None
        attn_output_209 = transpose_212.contiguous()
        transpose_212 = None
        reshape_158 = attn_output_209.reshape(1, 19, -1)
        attn_output_209 = None
        attn_output_210 = reshape_158.contiguous()
        reshape_158 = None
        attn_output_211 = torch._C._nn.linear(
            attn_output_210,
            l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_210 = l_self_modules_model_modules_layers_modules_52_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_525 = hidden_states_519 + attn_output_211
        hidden_states_519 = attn_output_211 = None
        hidden_states_526 = hidden_states_525.to(torch.float32)
        pow_106 = hidden_states_526.pow(2)
        variance_105 = pow_106.mean(-1, keepdim=True)
        pow_106 = None
        add_316 = variance_105 + 1e-05
        variance_105 = None
        rsqrt_105 = torch.rsqrt(add_316)
        add_316 = None
        hidden_states_527 = hidden_states_526 * rsqrt_105
        hidden_states_526 = rsqrt_105 = None
        to_215 = hidden_states_527.to(torch.float16)
        hidden_states_527 = None
        hidden_states_528 = (
            l_self_modules_model_modules_layers_modules_52_modules_post_attention_layernorm_parameters_weight_
            * to_215
        )
        l_self_modules_model_modules_layers_modules_52_modules_post_attention_layernorm_parameters_weight_ = (
            to_215
        ) = None
        linear_368 = torch._C._nn.linear(
            hidden_states_528,
            l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_52 = torch.nn.functional.silu(linear_368, inplace=False)
        linear_368 = None
        linear_369 = torch._C._nn.linear(
            hidden_states_528,
            l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_528 = l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_478 = silu_52 * linear_369
        silu_52 = linear_369 = None
        down_proj_52 = torch._C._nn.linear(
            mul_478,
            l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_478 = l_self_modules_model_modules_layers_modules_52_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_529 = hidden_states_525 + down_proj_52
        hidden_states_525 = down_proj_52 = None
        hidden_states_530 = hidden_states_529.to(torch.float32)
        pow_107 = hidden_states_530.pow(2)
        variance_106 = pow_107.mean(-1, keepdim=True)
        pow_107 = None
        add_318 = variance_106 + 1e-05
        variance_106 = None
        rsqrt_106 = torch.rsqrt(add_318)
        add_318 = None
        hidden_states_531 = hidden_states_530 * rsqrt_106
        hidden_states_530 = rsqrt_106 = None
        to_217 = hidden_states_531.to(torch.float16)
        hidden_states_531 = None
        hidden_states_532 = (
            l_self_modules_model_modules_layers_modules_53_modules_input_layernorm_parameters_weight_
            * to_217
        )
        l_self_modules_model_modules_layers_modules_53_modules_input_layernorm_parameters_weight_ = (
            to_217
        ) = None
        linear_371 = torch._C._nn.linear(
            hidden_states_532,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_159 = linear_371.view((1, 19, -1, 128))
        linear_371 = None
        query_states_53 = view_159.transpose(1, 2)
        view_159 = None
        linear_372 = torch._C._nn.linear(
            hidden_states_532,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_160 = linear_372.view((1, 19, -1, 128))
        linear_372 = None
        key_states_53 = view_160.transpose(1, 2)
        view_160 = None
        linear_373 = torch._C._nn.linear(
            hidden_states_532,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_532 = l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_161 = linear_373.view((1, 19, -1, 128))
        linear_373 = None
        value_states_53 = view_161.transpose(1, 2)
        view_161 = None
        cos_56 = cos_2.unsqueeze(1)
        sin_56 = sin_2.unsqueeze(1)
        mul_481 = query_states_53 * cos_56
        x1_106 = query_states_53[(Ellipsis, slice(None, 64, None))]
        x2_106 = query_states_53[(Ellipsis, slice(64, None, None))]
        query_states_53 = None
        neg_106 = -x2_106
        x2_106 = None
        cat_107 = torch.cat((neg_106, x1_106), dim=-1)
        neg_106 = x1_106 = None
        mul_482 = cat_107 * sin_56
        cat_107 = None
        q_embed_53 = mul_481 + mul_482
        mul_481 = mul_482 = None
        mul_483 = key_states_53 * cos_56
        cos_56 = None
        x1_107 = key_states_53[(Ellipsis, slice(None, 64, None))]
        x2_107 = key_states_53[(Ellipsis, slice(64, None, None))]
        key_states_53 = None
        neg_107 = -x2_107
        x2_107 = None
        cat_108 = torch.cat((neg_107, x1_107), dim=-1)
        neg_107 = x1_107 = None
        mul_484 = cat_108 * sin_56
        cat_108 = sin_56 = None
        k_embed_53 = mul_483 + mul_484
        mul_483 = mul_484 = None
        getitem_377 = k_embed_53[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_53 = None
        hidden_states_533 = getitem_377.expand(1, 8, 5, 19, 128)
        getitem_377 = None
        key_106 = hidden_states_533.reshape(1, 40, 19, 128)
        hidden_states_533 = None
        getitem_378 = value_states_53[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_53 = None
        hidden_states_534 = getitem_378.expand(1, 8, 5, 19, 128)
        getitem_378 = None
        value_106 = hidden_states_534.reshape(1, 40, 19, 128)
        hidden_states_534 = None
        attention_mask_54 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_53 = q_embed_53.contiguous()
        q_embed_53 = None
        key_107 = key_106.contiguous()
        key_106 = None
        value_107 = value_106.contiguous()
        value_106 = None
        attn_output_212 = torch._C._nn.scaled_dot_product_attention(
            query_53,
            key_107,
            value_107,
            attn_mask=attention_mask_54,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_53 = key_107 = value_107 = attention_mask_54 = None
        transpose_216 = attn_output_212.transpose(1, 2)
        attn_output_212 = None
        attn_output_213 = transpose_216.contiguous()
        transpose_216 = None
        reshape_161 = attn_output_213.reshape(1, 19, -1)
        attn_output_213 = None
        attn_output_214 = reshape_161.contiguous()
        reshape_161 = None
        attn_output_215 = torch._C._nn.linear(
            attn_output_214,
            l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_214 = l_self_modules_model_modules_layers_modules_53_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_535 = hidden_states_529 + attn_output_215
        hidden_states_529 = attn_output_215 = None
        hidden_states_536 = hidden_states_535.to(torch.float32)
        pow_108 = hidden_states_536.pow(2)
        variance_107 = pow_108.mean(-1, keepdim=True)
        pow_108 = None
        add_322 = variance_107 + 1e-05
        variance_107 = None
        rsqrt_107 = torch.rsqrt(add_322)
        add_322 = None
        hidden_states_537 = hidden_states_536 * rsqrt_107
        hidden_states_536 = rsqrt_107 = None
        to_219 = hidden_states_537.to(torch.float16)
        hidden_states_537 = None
        hidden_states_538 = (
            l_self_modules_model_modules_layers_modules_53_modules_post_attention_layernorm_parameters_weight_
            * to_219
        )
        l_self_modules_model_modules_layers_modules_53_modules_post_attention_layernorm_parameters_weight_ = (
            to_219
        ) = None
        linear_375 = torch._C._nn.linear(
            hidden_states_538,
            l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_53 = torch.nn.functional.silu(linear_375, inplace=False)
        linear_375 = None
        linear_376 = torch._C._nn.linear(
            hidden_states_538,
            l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_538 = l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_487 = silu_53 * linear_376
        silu_53 = linear_376 = None
        down_proj_53 = torch._C._nn.linear(
            mul_487,
            l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_487 = l_self_modules_model_modules_layers_modules_53_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_539 = hidden_states_535 + down_proj_53
        hidden_states_535 = down_proj_53 = None
        hidden_states_540 = hidden_states_539.to(torch.float32)
        pow_109 = hidden_states_540.pow(2)
        variance_108 = pow_109.mean(-1, keepdim=True)
        pow_109 = None
        add_324 = variance_108 + 1e-05
        variance_108 = None
        rsqrt_108 = torch.rsqrt(add_324)
        add_324 = None
        hidden_states_541 = hidden_states_540 * rsqrt_108
        hidden_states_540 = rsqrt_108 = None
        to_221 = hidden_states_541.to(torch.float16)
        hidden_states_541 = None
        hidden_states_542 = (
            l_self_modules_model_modules_layers_modules_54_modules_input_layernorm_parameters_weight_
            * to_221
        )
        l_self_modules_model_modules_layers_modules_54_modules_input_layernorm_parameters_weight_ = (
            to_221
        ) = None
        linear_378 = torch._C._nn.linear(
            hidden_states_542,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_162 = linear_378.view((1, 19, -1, 128))
        linear_378 = None
        query_states_54 = view_162.transpose(1, 2)
        view_162 = None
        linear_379 = torch._C._nn.linear(
            hidden_states_542,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_163 = linear_379.view((1, 19, -1, 128))
        linear_379 = None
        key_states_54 = view_163.transpose(1, 2)
        view_163 = None
        linear_380 = torch._C._nn.linear(
            hidden_states_542,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_542 = l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_164 = linear_380.view((1, 19, -1, 128))
        linear_380 = None
        value_states_54 = view_164.transpose(1, 2)
        view_164 = None
        cos_57 = cos_2.unsqueeze(1)
        sin_57 = sin_2.unsqueeze(1)
        mul_490 = query_states_54 * cos_57
        x1_108 = query_states_54[(Ellipsis, slice(None, 64, None))]
        x2_108 = query_states_54[(Ellipsis, slice(64, None, None))]
        query_states_54 = None
        neg_108 = -x2_108
        x2_108 = None
        cat_109 = torch.cat((neg_108, x1_108), dim=-1)
        neg_108 = x1_108 = None
        mul_491 = cat_109 * sin_57
        cat_109 = None
        q_embed_54 = mul_490 + mul_491
        mul_490 = mul_491 = None
        mul_492 = key_states_54 * cos_57
        cos_57 = None
        x1_109 = key_states_54[(Ellipsis, slice(None, 64, None))]
        x2_109 = key_states_54[(Ellipsis, slice(64, None, None))]
        key_states_54 = None
        neg_109 = -x2_109
        x2_109 = None
        cat_110 = torch.cat((neg_109, x1_109), dim=-1)
        neg_109 = x1_109 = None
        mul_493 = cat_110 * sin_57
        cat_110 = sin_57 = None
        k_embed_54 = mul_492 + mul_493
        mul_492 = mul_493 = None
        getitem_384 = k_embed_54[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_54 = None
        hidden_states_543 = getitem_384.expand(1, 8, 5, 19, 128)
        getitem_384 = None
        key_108 = hidden_states_543.reshape(1, 40, 19, 128)
        hidden_states_543 = None
        getitem_385 = value_states_54[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_54 = None
        hidden_states_544 = getitem_385.expand(1, 8, 5, 19, 128)
        getitem_385 = None
        value_108 = hidden_states_544.reshape(1, 40, 19, 128)
        hidden_states_544 = None
        attention_mask_55 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_54 = q_embed_54.contiguous()
        q_embed_54 = None
        key_109 = key_108.contiguous()
        key_108 = None
        value_109 = value_108.contiguous()
        value_108 = None
        attn_output_216 = torch._C._nn.scaled_dot_product_attention(
            query_54,
            key_109,
            value_109,
            attn_mask=attention_mask_55,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_54 = key_109 = value_109 = attention_mask_55 = None
        transpose_220 = attn_output_216.transpose(1, 2)
        attn_output_216 = None
        attn_output_217 = transpose_220.contiguous()
        transpose_220 = None
        reshape_164 = attn_output_217.reshape(1, 19, -1)
        attn_output_217 = None
        attn_output_218 = reshape_164.contiguous()
        reshape_164 = None
        attn_output_219 = torch._C._nn.linear(
            attn_output_218,
            l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_218 = l_self_modules_model_modules_layers_modules_54_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_545 = hidden_states_539 + attn_output_219
        hidden_states_539 = attn_output_219 = None
        hidden_states_546 = hidden_states_545.to(torch.float32)
        pow_110 = hidden_states_546.pow(2)
        variance_109 = pow_110.mean(-1, keepdim=True)
        pow_110 = None
        add_328 = variance_109 + 1e-05
        variance_109 = None
        rsqrt_109 = torch.rsqrt(add_328)
        add_328 = None
        hidden_states_547 = hidden_states_546 * rsqrt_109
        hidden_states_546 = rsqrt_109 = None
        to_223 = hidden_states_547.to(torch.float16)
        hidden_states_547 = None
        hidden_states_548 = (
            l_self_modules_model_modules_layers_modules_54_modules_post_attention_layernorm_parameters_weight_
            * to_223
        )
        l_self_modules_model_modules_layers_modules_54_modules_post_attention_layernorm_parameters_weight_ = (
            to_223
        ) = None
        linear_382 = torch._C._nn.linear(
            hidden_states_548,
            l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_54 = torch.nn.functional.silu(linear_382, inplace=False)
        linear_382 = None
        linear_383 = torch._C._nn.linear(
            hidden_states_548,
            l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_548 = l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_496 = silu_54 * linear_383
        silu_54 = linear_383 = None
        down_proj_54 = torch._C._nn.linear(
            mul_496,
            l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_496 = l_self_modules_model_modules_layers_modules_54_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_549 = hidden_states_545 + down_proj_54
        hidden_states_545 = down_proj_54 = None
        hidden_states_550 = hidden_states_549.to(torch.float32)
        pow_111 = hidden_states_550.pow(2)
        variance_110 = pow_111.mean(-1, keepdim=True)
        pow_111 = None
        add_330 = variance_110 + 1e-05
        variance_110 = None
        rsqrt_110 = torch.rsqrt(add_330)
        add_330 = None
        hidden_states_551 = hidden_states_550 * rsqrt_110
        hidden_states_550 = rsqrt_110 = None
        to_225 = hidden_states_551.to(torch.float16)
        hidden_states_551 = None
        hidden_states_552 = (
            l_self_modules_model_modules_layers_modules_55_modules_input_layernorm_parameters_weight_
            * to_225
        )
        l_self_modules_model_modules_layers_modules_55_modules_input_layernorm_parameters_weight_ = (
            to_225
        ) = None
        linear_385 = torch._C._nn.linear(
            hidden_states_552,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_165 = linear_385.view((1, 19, -1, 128))
        linear_385 = None
        query_states_55 = view_165.transpose(1, 2)
        view_165 = None
        linear_386 = torch._C._nn.linear(
            hidden_states_552,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_166 = linear_386.view((1, 19, -1, 128))
        linear_386 = None
        key_states_55 = view_166.transpose(1, 2)
        view_166 = None
        linear_387 = torch._C._nn.linear(
            hidden_states_552,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_552 = l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_167 = linear_387.view((1, 19, -1, 128))
        linear_387 = None
        value_states_55 = view_167.transpose(1, 2)
        view_167 = None
        cos_58 = cos_2.unsqueeze(1)
        sin_58 = sin_2.unsqueeze(1)
        mul_499 = query_states_55 * cos_58
        x1_110 = query_states_55[(Ellipsis, slice(None, 64, None))]
        x2_110 = query_states_55[(Ellipsis, slice(64, None, None))]
        query_states_55 = None
        neg_110 = -x2_110
        x2_110 = None
        cat_111 = torch.cat((neg_110, x1_110), dim=-1)
        neg_110 = x1_110 = None
        mul_500 = cat_111 * sin_58
        cat_111 = None
        q_embed_55 = mul_499 + mul_500
        mul_499 = mul_500 = None
        mul_501 = key_states_55 * cos_58
        cos_58 = None
        x1_111 = key_states_55[(Ellipsis, slice(None, 64, None))]
        x2_111 = key_states_55[(Ellipsis, slice(64, None, None))]
        key_states_55 = None
        neg_111 = -x2_111
        x2_111 = None
        cat_112 = torch.cat((neg_111, x1_111), dim=-1)
        neg_111 = x1_111 = None
        mul_502 = cat_112 * sin_58
        cat_112 = sin_58 = None
        k_embed_55 = mul_501 + mul_502
        mul_501 = mul_502 = None
        getitem_391 = k_embed_55[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_55 = None
        hidden_states_553 = getitem_391.expand(1, 8, 5, 19, 128)
        getitem_391 = None
        key_110 = hidden_states_553.reshape(1, 40, 19, 128)
        hidden_states_553 = None
        getitem_392 = value_states_55[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_55 = None
        hidden_states_554 = getitem_392.expand(1, 8, 5, 19, 128)
        getitem_392 = None
        value_110 = hidden_states_554.reshape(1, 40, 19, 128)
        hidden_states_554 = None
        attention_mask_56 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_55 = q_embed_55.contiguous()
        q_embed_55 = None
        key_111 = key_110.contiguous()
        key_110 = None
        value_111 = value_110.contiguous()
        value_110 = None
        attn_output_220 = torch._C._nn.scaled_dot_product_attention(
            query_55,
            key_111,
            value_111,
            attn_mask=attention_mask_56,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_55 = key_111 = value_111 = attention_mask_56 = None
        transpose_224 = attn_output_220.transpose(1, 2)
        attn_output_220 = None
        attn_output_221 = transpose_224.contiguous()
        transpose_224 = None
        reshape_167 = attn_output_221.reshape(1, 19, -1)
        attn_output_221 = None
        attn_output_222 = reshape_167.contiguous()
        reshape_167 = None
        attn_output_223 = torch._C._nn.linear(
            attn_output_222,
            l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_222 = l_self_modules_model_modules_layers_modules_55_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_555 = hidden_states_549 + attn_output_223
        hidden_states_549 = attn_output_223 = None
        hidden_states_556 = hidden_states_555.to(torch.float32)
        pow_112 = hidden_states_556.pow(2)
        variance_111 = pow_112.mean(-1, keepdim=True)
        pow_112 = None
        add_334 = variance_111 + 1e-05
        variance_111 = None
        rsqrt_111 = torch.rsqrt(add_334)
        add_334 = None
        hidden_states_557 = hidden_states_556 * rsqrt_111
        hidden_states_556 = rsqrt_111 = None
        to_227 = hidden_states_557.to(torch.float16)
        hidden_states_557 = None
        hidden_states_558 = (
            l_self_modules_model_modules_layers_modules_55_modules_post_attention_layernorm_parameters_weight_
            * to_227
        )
        l_self_modules_model_modules_layers_modules_55_modules_post_attention_layernorm_parameters_weight_ = (
            to_227
        ) = None
        linear_389 = torch._C._nn.linear(
            hidden_states_558,
            l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_55 = torch.nn.functional.silu(linear_389, inplace=False)
        linear_389 = None
        linear_390 = torch._C._nn.linear(
            hidden_states_558,
            l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_558 = l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_505 = silu_55 * linear_390
        silu_55 = linear_390 = None
        down_proj_55 = torch._C._nn.linear(
            mul_505,
            l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_505 = l_self_modules_model_modules_layers_modules_55_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_559 = hidden_states_555 + down_proj_55
        hidden_states_555 = down_proj_55 = None
        hidden_states_560 = hidden_states_559.to(torch.float32)
        pow_113 = hidden_states_560.pow(2)
        variance_112 = pow_113.mean(-1, keepdim=True)
        pow_113 = None
        add_336 = variance_112 + 1e-05
        variance_112 = None
        rsqrt_112 = torch.rsqrt(add_336)
        add_336 = None
        hidden_states_561 = hidden_states_560 * rsqrt_112
        hidden_states_560 = rsqrt_112 = None
        to_229 = hidden_states_561.to(torch.float16)
        hidden_states_561 = None
        hidden_states_562 = (
            l_self_modules_model_modules_layers_modules_56_modules_input_layernorm_parameters_weight_
            * to_229
        )
        l_self_modules_model_modules_layers_modules_56_modules_input_layernorm_parameters_weight_ = (
            to_229
        ) = None
        linear_392 = torch._C._nn.linear(
            hidden_states_562,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_168 = linear_392.view((1, 19, -1, 128))
        linear_392 = None
        query_states_56 = view_168.transpose(1, 2)
        view_168 = None
        linear_393 = torch._C._nn.linear(
            hidden_states_562,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_169 = linear_393.view((1, 19, -1, 128))
        linear_393 = None
        key_states_56 = view_169.transpose(1, 2)
        view_169 = None
        linear_394 = torch._C._nn.linear(
            hidden_states_562,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_562 = l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_170 = linear_394.view((1, 19, -1, 128))
        linear_394 = None
        value_states_56 = view_170.transpose(1, 2)
        view_170 = None
        cos_59 = cos_2.unsqueeze(1)
        sin_59 = sin_2.unsqueeze(1)
        mul_508 = query_states_56 * cos_59
        x1_112 = query_states_56[(Ellipsis, slice(None, 64, None))]
        x2_112 = query_states_56[(Ellipsis, slice(64, None, None))]
        query_states_56 = None
        neg_112 = -x2_112
        x2_112 = None
        cat_113 = torch.cat((neg_112, x1_112), dim=-1)
        neg_112 = x1_112 = None
        mul_509 = cat_113 * sin_59
        cat_113 = None
        q_embed_56 = mul_508 + mul_509
        mul_508 = mul_509 = None
        mul_510 = key_states_56 * cos_59
        cos_59 = None
        x1_113 = key_states_56[(Ellipsis, slice(None, 64, None))]
        x2_113 = key_states_56[(Ellipsis, slice(64, None, None))]
        key_states_56 = None
        neg_113 = -x2_113
        x2_113 = None
        cat_114 = torch.cat((neg_113, x1_113), dim=-1)
        neg_113 = x1_113 = None
        mul_511 = cat_114 * sin_59
        cat_114 = sin_59 = None
        k_embed_56 = mul_510 + mul_511
        mul_510 = mul_511 = None
        getitem_398 = k_embed_56[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_56 = None
        hidden_states_563 = getitem_398.expand(1, 8, 5, 19, 128)
        getitem_398 = None
        key_112 = hidden_states_563.reshape(1, 40, 19, 128)
        hidden_states_563 = None
        getitem_399 = value_states_56[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_56 = None
        hidden_states_564 = getitem_399.expand(1, 8, 5, 19, 128)
        getitem_399 = None
        value_112 = hidden_states_564.reshape(1, 40, 19, 128)
        hidden_states_564 = None
        attention_mask_57 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_56 = q_embed_56.contiguous()
        q_embed_56 = None
        key_113 = key_112.contiguous()
        key_112 = None
        value_113 = value_112.contiguous()
        value_112 = None
        attn_output_224 = torch._C._nn.scaled_dot_product_attention(
            query_56,
            key_113,
            value_113,
            attn_mask=attention_mask_57,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_56 = key_113 = value_113 = attention_mask_57 = None
        transpose_228 = attn_output_224.transpose(1, 2)
        attn_output_224 = None
        attn_output_225 = transpose_228.contiguous()
        transpose_228 = None
        reshape_170 = attn_output_225.reshape(1, 19, -1)
        attn_output_225 = None
        attn_output_226 = reshape_170.contiguous()
        reshape_170 = None
        attn_output_227 = torch._C._nn.linear(
            attn_output_226,
            l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_226 = l_self_modules_model_modules_layers_modules_56_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_565 = hidden_states_559 + attn_output_227
        hidden_states_559 = attn_output_227 = None
        hidden_states_566 = hidden_states_565.to(torch.float32)
        pow_114 = hidden_states_566.pow(2)
        variance_113 = pow_114.mean(-1, keepdim=True)
        pow_114 = None
        add_340 = variance_113 + 1e-05
        variance_113 = None
        rsqrt_113 = torch.rsqrt(add_340)
        add_340 = None
        hidden_states_567 = hidden_states_566 * rsqrt_113
        hidden_states_566 = rsqrt_113 = None
        to_231 = hidden_states_567.to(torch.float16)
        hidden_states_567 = None
        hidden_states_568 = (
            l_self_modules_model_modules_layers_modules_56_modules_post_attention_layernorm_parameters_weight_
            * to_231
        )
        l_self_modules_model_modules_layers_modules_56_modules_post_attention_layernorm_parameters_weight_ = (
            to_231
        ) = None
        linear_396 = torch._C._nn.linear(
            hidden_states_568,
            l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_56 = torch.nn.functional.silu(linear_396, inplace=False)
        linear_396 = None
        linear_397 = torch._C._nn.linear(
            hidden_states_568,
            l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_568 = l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_514 = silu_56 * linear_397
        silu_56 = linear_397 = None
        down_proj_56 = torch._C._nn.linear(
            mul_514,
            l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_514 = l_self_modules_model_modules_layers_modules_56_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_569 = hidden_states_565 + down_proj_56
        hidden_states_565 = down_proj_56 = None
        hidden_states_570 = hidden_states_569.to(torch.float32)
        pow_115 = hidden_states_570.pow(2)
        variance_114 = pow_115.mean(-1, keepdim=True)
        pow_115 = None
        add_342 = variance_114 + 1e-05
        variance_114 = None
        rsqrt_114 = torch.rsqrt(add_342)
        add_342 = None
        hidden_states_571 = hidden_states_570 * rsqrt_114
        hidden_states_570 = rsqrt_114 = None
        to_233 = hidden_states_571.to(torch.float16)
        hidden_states_571 = None
        hidden_states_572 = (
            l_self_modules_model_modules_layers_modules_57_modules_input_layernorm_parameters_weight_
            * to_233
        )
        l_self_modules_model_modules_layers_modules_57_modules_input_layernorm_parameters_weight_ = (
            to_233
        ) = None
        linear_399 = torch._C._nn.linear(
            hidden_states_572,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_171 = linear_399.view((1, 19, -1, 128))
        linear_399 = None
        query_states_57 = view_171.transpose(1, 2)
        view_171 = None
        linear_400 = torch._C._nn.linear(
            hidden_states_572,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_172 = linear_400.view((1, 19, -1, 128))
        linear_400 = None
        key_states_57 = view_172.transpose(1, 2)
        view_172 = None
        linear_401 = torch._C._nn.linear(
            hidden_states_572,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_572 = l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_173 = linear_401.view((1, 19, -1, 128))
        linear_401 = None
        value_states_57 = view_173.transpose(1, 2)
        view_173 = None
        cos_60 = cos_2.unsqueeze(1)
        sin_60 = sin_2.unsqueeze(1)
        mul_517 = query_states_57 * cos_60
        x1_114 = query_states_57[(Ellipsis, slice(None, 64, None))]
        x2_114 = query_states_57[(Ellipsis, slice(64, None, None))]
        query_states_57 = None
        neg_114 = -x2_114
        x2_114 = None
        cat_115 = torch.cat((neg_114, x1_114), dim=-1)
        neg_114 = x1_114 = None
        mul_518 = cat_115 * sin_60
        cat_115 = None
        q_embed_57 = mul_517 + mul_518
        mul_517 = mul_518 = None
        mul_519 = key_states_57 * cos_60
        cos_60 = None
        x1_115 = key_states_57[(Ellipsis, slice(None, 64, None))]
        x2_115 = key_states_57[(Ellipsis, slice(64, None, None))]
        key_states_57 = None
        neg_115 = -x2_115
        x2_115 = None
        cat_116 = torch.cat((neg_115, x1_115), dim=-1)
        neg_115 = x1_115 = None
        mul_520 = cat_116 * sin_60
        cat_116 = sin_60 = None
        k_embed_57 = mul_519 + mul_520
        mul_519 = mul_520 = None
        getitem_405 = k_embed_57[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_57 = None
        hidden_states_573 = getitem_405.expand(1, 8, 5, 19, 128)
        getitem_405 = None
        key_114 = hidden_states_573.reshape(1, 40, 19, 128)
        hidden_states_573 = None
        getitem_406 = value_states_57[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_57 = None
        hidden_states_574 = getitem_406.expand(1, 8, 5, 19, 128)
        getitem_406 = None
        value_114 = hidden_states_574.reshape(1, 40, 19, 128)
        hidden_states_574 = None
        attention_mask_58 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_57 = q_embed_57.contiguous()
        q_embed_57 = None
        key_115 = key_114.contiguous()
        key_114 = None
        value_115 = value_114.contiguous()
        value_114 = None
        attn_output_228 = torch._C._nn.scaled_dot_product_attention(
            query_57,
            key_115,
            value_115,
            attn_mask=attention_mask_58,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_57 = key_115 = value_115 = attention_mask_58 = None
        transpose_232 = attn_output_228.transpose(1, 2)
        attn_output_228 = None
        attn_output_229 = transpose_232.contiguous()
        transpose_232 = None
        reshape_173 = attn_output_229.reshape(1, 19, -1)
        attn_output_229 = None
        attn_output_230 = reshape_173.contiguous()
        reshape_173 = None
        attn_output_231 = torch._C._nn.linear(
            attn_output_230,
            l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_230 = l_self_modules_model_modules_layers_modules_57_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_575 = hidden_states_569 + attn_output_231
        hidden_states_569 = attn_output_231 = None
        hidden_states_576 = hidden_states_575.to(torch.float32)
        pow_116 = hidden_states_576.pow(2)
        variance_115 = pow_116.mean(-1, keepdim=True)
        pow_116 = None
        add_346 = variance_115 + 1e-05
        variance_115 = None
        rsqrt_115 = torch.rsqrt(add_346)
        add_346 = None
        hidden_states_577 = hidden_states_576 * rsqrt_115
        hidden_states_576 = rsqrt_115 = None
        to_235 = hidden_states_577.to(torch.float16)
        hidden_states_577 = None
        hidden_states_578 = (
            l_self_modules_model_modules_layers_modules_57_modules_post_attention_layernorm_parameters_weight_
            * to_235
        )
        l_self_modules_model_modules_layers_modules_57_modules_post_attention_layernorm_parameters_weight_ = (
            to_235
        ) = None
        linear_403 = torch._C._nn.linear(
            hidden_states_578,
            l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_57 = torch.nn.functional.silu(linear_403, inplace=False)
        linear_403 = None
        linear_404 = torch._C._nn.linear(
            hidden_states_578,
            l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_578 = l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_523 = silu_57 * linear_404
        silu_57 = linear_404 = None
        down_proj_57 = torch._C._nn.linear(
            mul_523,
            l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_523 = l_self_modules_model_modules_layers_modules_57_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_579 = hidden_states_575 + down_proj_57
        hidden_states_575 = down_proj_57 = None
        hidden_states_580 = hidden_states_579.to(torch.float32)
        pow_117 = hidden_states_580.pow(2)
        variance_116 = pow_117.mean(-1, keepdim=True)
        pow_117 = None
        add_348 = variance_116 + 1e-05
        variance_116 = None
        rsqrt_116 = torch.rsqrt(add_348)
        add_348 = None
        hidden_states_581 = hidden_states_580 * rsqrt_116
        hidden_states_580 = rsqrt_116 = None
        to_237 = hidden_states_581.to(torch.float16)
        hidden_states_581 = None
        hidden_states_582 = (
            l_self_modules_model_modules_layers_modules_58_modules_input_layernorm_parameters_weight_
            * to_237
        )
        l_self_modules_model_modules_layers_modules_58_modules_input_layernorm_parameters_weight_ = (
            to_237
        ) = None
        linear_406 = torch._C._nn.linear(
            hidden_states_582,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_174 = linear_406.view((1, 19, -1, 128))
        linear_406 = None
        query_states_58 = view_174.transpose(1, 2)
        view_174 = None
        linear_407 = torch._C._nn.linear(
            hidden_states_582,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_175 = linear_407.view((1, 19, -1, 128))
        linear_407 = None
        key_states_58 = view_175.transpose(1, 2)
        view_175 = None
        linear_408 = torch._C._nn.linear(
            hidden_states_582,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_582 = l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_176 = linear_408.view((1, 19, -1, 128))
        linear_408 = None
        value_states_58 = view_176.transpose(1, 2)
        view_176 = None
        cos_61 = cos_2.unsqueeze(1)
        sin_61 = sin_2.unsqueeze(1)
        mul_526 = query_states_58 * cos_61
        x1_116 = query_states_58[(Ellipsis, slice(None, 64, None))]
        x2_116 = query_states_58[(Ellipsis, slice(64, None, None))]
        query_states_58 = None
        neg_116 = -x2_116
        x2_116 = None
        cat_117 = torch.cat((neg_116, x1_116), dim=-1)
        neg_116 = x1_116 = None
        mul_527 = cat_117 * sin_61
        cat_117 = None
        q_embed_58 = mul_526 + mul_527
        mul_526 = mul_527 = None
        mul_528 = key_states_58 * cos_61
        cos_61 = None
        x1_117 = key_states_58[(Ellipsis, slice(None, 64, None))]
        x2_117 = key_states_58[(Ellipsis, slice(64, None, None))]
        key_states_58 = None
        neg_117 = -x2_117
        x2_117 = None
        cat_118 = torch.cat((neg_117, x1_117), dim=-1)
        neg_117 = x1_117 = None
        mul_529 = cat_118 * sin_61
        cat_118 = sin_61 = None
        k_embed_58 = mul_528 + mul_529
        mul_528 = mul_529 = None
        getitem_412 = k_embed_58[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_58 = None
        hidden_states_583 = getitem_412.expand(1, 8, 5, 19, 128)
        getitem_412 = None
        key_116 = hidden_states_583.reshape(1, 40, 19, 128)
        hidden_states_583 = None
        getitem_413 = value_states_58[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_58 = None
        hidden_states_584 = getitem_413.expand(1, 8, 5, 19, 128)
        getitem_413 = None
        value_116 = hidden_states_584.reshape(1, 40, 19, 128)
        hidden_states_584 = None
        attention_mask_59 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_58 = q_embed_58.contiguous()
        q_embed_58 = None
        key_117 = key_116.contiguous()
        key_116 = None
        value_117 = value_116.contiguous()
        value_116 = None
        attn_output_232 = torch._C._nn.scaled_dot_product_attention(
            query_58,
            key_117,
            value_117,
            attn_mask=attention_mask_59,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_58 = key_117 = value_117 = attention_mask_59 = None
        transpose_236 = attn_output_232.transpose(1, 2)
        attn_output_232 = None
        attn_output_233 = transpose_236.contiguous()
        transpose_236 = None
        reshape_176 = attn_output_233.reshape(1, 19, -1)
        attn_output_233 = None
        attn_output_234 = reshape_176.contiguous()
        reshape_176 = None
        attn_output_235 = torch._C._nn.linear(
            attn_output_234,
            l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_234 = l_self_modules_model_modules_layers_modules_58_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_585 = hidden_states_579 + attn_output_235
        hidden_states_579 = attn_output_235 = None
        hidden_states_586 = hidden_states_585.to(torch.float32)
        pow_118 = hidden_states_586.pow(2)
        variance_117 = pow_118.mean(-1, keepdim=True)
        pow_118 = None
        add_352 = variance_117 + 1e-05
        variance_117 = None
        rsqrt_117 = torch.rsqrt(add_352)
        add_352 = None
        hidden_states_587 = hidden_states_586 * rsqrt_117
        hidden_states_586 = rsqrt_117 = None
        to_239 = hidden_states_587.to(torch.float16)
        hidden_states_587 = None
        hidden_states_588 = (
            l_self_modules_model_modules_layers_modules_58_modules_post_attention_layernorm_parameters_weight_
            * to_239
        )
        l_self_modules_model_modules_layers_modules_58_modules_post_attention_layernorm_parameters_weight_ = (
            to_239
        ) = None
        linear_410 = torch._C._nn.linear(
            hidden_states_588,
            l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_58 = torch.nn.functional.silu(linear_410, inplace=False)
        linear_410 = None
        linear_411 = torch._C._nn.linear(
            hidden_states_588,
            l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_588 = l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_532 = silu_58 * linear_411
        silu_58 = linear_411 = None
        down_proj_58 = torch._C._nn.linear(
            mul_532,
            l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_532 = l_self_modules_model_modules_layers_modules_58_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_589 = hidden_states_585 + down_proj_58
        hidden_states_585 = down_proj_58 = None
        hidden_states_590 = hidden_states_589.to(torch.float32)
        pow_119 = hidden_states_590.pow(2)
        variance_118 = pow_119.mean(-1, keepdim=True)
        pow_119 = None
        add_354 = variance_118 + 1e-05
        variance_118 = None
        rsqrt_118 = torch.rsqrt(add_354)
        add_354 = None
        hidden_states_591 = hidden_states_590 * rsqrt_118
        hidden_states_590 = rsqrt_118 = None
        to_241 = hidden_states_591.to(torch.float16)
        hidden_states_591 = None
        hidden_states_592 = (
            l_self_modules_model_modules_layers_modules_59_modules_input_layernorm_parameters_weight_
            * to_241
        )
        l_self_modules_model_modules_layers_modules_59_modules_input_layernorm_parameters_weight_ = (
            to_241
        ) = None
        linear_413 = torch._C._nn.linear(
            hidden_states_592,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_177 = linear_413.view((1, 19, -1, 128))
        linear_413 = None
        query_states_59 = view_177.transpose(1, 2)
        view_177 = None
        linear_414 = torch._C._nn.linear(
            hidden_states_592,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_178 = linear_414.view((1, 19, -1, 128))
        linear_414 = None
        key_states_59 = view_178.transpose(1, 2)
        view_178 = None
        linear_415 = torch._C._nn.linear(
            hidden_states_592,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_592 = l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_179 = linear_415.view((1, 19, -1, 128))
        linear_415 = None
        value_states_59 = view_179.transpose(1, 2)
        view_179 = None
        cos_62 = cos_2.unsqueeze(1)
        sin_62 = sin_2.unsqueeze(1)
        mul_535 = query_states_59 * cos_62
        x1_118 = query_states_59[(Ellipsis, slice(None, 64, None))]
        x2_118 = query_states_59[(Ellipsis, slice(64, None, None))]
        query_states_59 = None
        neg_118 = -x2_118
        x2_118 = None
        cat_119 = torch.cat((neg_118, x1_118), dim=-1)
        neg_118 = x1_118 = None
        mul_536 = cat_119 * sin_62
        cat_119 = None
        q_embed_59 = mul_535 + mul_536
        mul_535 = mul_536 = None
        mul_537 = key_states_59 * cos_62
        cos_62 = None
        x1_119 = key_states_59[(Ellipsis, slice(None, 64, None))]
        x2_119 = key_states_59[(Ellipsis, slice(64, None, None))]
        key_states_59 = None
        neg_119 = -x2_119
        x2_119 = None
        cat_120 = torch.cat((neg_119, x1_119), dim=-1)
        neg_119 = x1_119 = None
        mul_538 = cat_120 * sin_62
        cat_120 = sin_62 = None
        k_embed_59 = mul_537 + mul_538
        mul_537 = mul_538 = None
        getitem_419 = k_embed_59[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_59 = None
        hidden_states_593 = getitem_419.expand(1, 8, 5, 19, 128)
        getitem_419 = None
        key_118 = hidden_states_593.reshape(1, 40, 19, 128)
        hidden_states_593 = None
        getitem_420 = value_states_59[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_59 = None
        hidden_states_594 = getitem_420.expand(1, 8, 5, 19, 128)
        getitem_420 = None
        value_118 = hidden_states_594.reshape(1, 40, 19, 128)
        hidden_states_594 = None
        attention_mask_60 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_59 = q_embed_59.contiguous()
        q_embed_59 = None
        key_119 = key_118.contiguous()
        key_118 = None
        value_119 = value_118.contiguous()
        value_118 = None
        attn_output_236 = torch._C._nn.scaled_dot_product_attention(
            query_59,
            key_119,
            value_119,
            attn_mask=attention_mask_60,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_59 = key_119 = value_119 = attention_mask_60 = None
        transpose_240 = attn_output_236.transpose(1, 2)
        attn_output_236 = None
        attn_output_237 = transpose_240.contiguous()
        transpose_240 = None
        reshape_179 = attn_output_237.reshape(1, 19, -1)
        attn_output_237 = None
        attn_output_238 = reshape_179.contiguous()
        reshape_179 = None
        attn_output_239 = torch._C._nn.linear(
            attn_output_238,
            l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_238 = l_self_modules_model_modules_layers_modules_59_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_595 = hidden_states_589 + attn_output_239
        hidden_states_589 = attn_output_239 = None
        hidden_states_596 = hidden_states_595.to(torch.float32)
        pow_120 = hidden_states_596.pow(2)
        variance_119 = pow_120.mean(-1, keepdim=True)
        pow_120 = None
        add_358 = variance_119 + 1e-05
        variance_119 = None
        rsqrt_119 = torch.rsqrt(add_358)
        add_358 = None
        hidden_states_597 = hidden_states_596 * rsqrt_119
        hidden_states_596 = rsqrt_119 = None
        to_243 = hidden_states_597.to(torch.float16)
        hidden_states_597 = None
        hidden_states_598 = (
            l_self_modules_model_modules_layers_modules_59_modules_post_attention_layernorm_parameters_weight_
            * to_243
        )
        l_self_modules_model_modules_layers_modules_59_modules_post_attention_layernorm_parameters_weight_ = (
            to_243
        ) = None
        linear_417 = torch._C._nn.linear(
            hidden_states_598,
            l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_59 = torch.nn.functional.silu(linear_417, inplace=False)
        linear_417 = None
        linear_418 = torch._C._nn.linear(
            hidden_states_598,
            l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_598 = l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_541 = silu_59 * linear_418
        silu_59 = linear_418 = None
        down_proj_59 = torch._C._nn.linear(
            mul_541,
            l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_541 = l_self_modules_model_modules_layers_modules_59_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_599 = hidden_states_595 + down_proj_59
        hidden_states_595 = down_proj_59 = None
        hidden_states_600 = hidden_states_599.to(torch.float32)
        pow_121 = hidden_states_600.pow(2)
        variance_120 = pow_121.mean(-1, keepdim=True)
        pow_121 = None
        add_360 = variance_120 + 1e-05
        variance_120 = None
        rsqrt_120 = torch.rsqrt(add_360)
        add_360 = None
        hidden_states_601 = hidden_states_600 * rsqrt_120
        hidden_states_600 = rsqrt_120 = None
        to_245 = hidden_states_601.to(torch.float16)
        hidden_states_601 = None
        hidden_states_602 = (
            l_self_modules_model_modules_layers_modules_60_modules_input_layernorm_parameters_weight_
            * to_245
        )
        l_self_modules_model_modules_layers_modules_60_modules_input_layernorm_parameters_weight_ = (
            to_245
        ) = None
        linear_420 = torch._C._nn.linear(
            hidden_states_602,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_180 = linear_420.view((1, 19, -1, 128))
        linear_420 = None
        query_states_60 = view_180.transpose(1, 2)
        view_180 = None
        linear_421 = torch._C._nn.linear(
            hidden_states_602,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_181 = linear_421.view((1, 19, -1, 128))
        linear_421 = None
        key_states_60 = view_181.transpose(1, 2)
        view_181 = None
        linear_422 = torch._C._nn.linear(
            hidden_states_602,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_602 = l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_182 = linear_422.view((1, 19, -1, 128))
        linear_422 = None
        value_states_60 = view_182.transpose(1, 2)
        view_182 = None
        cos_63 = cos_2.unsqueeze(1)
        sin_63 = sin_2.unsqueeze(1)
        mul_544 = query_states_60 * cos_63
        x1_120 = query_states_60[(Ellipsis, slice(None, 64, None))]
        x2_120 = query_states_60[(Ellipsis, slice(64, None, None))]
        query_states_60 = None
        neg_120 = -x2_120
        x2_120 = None
        cat_121 = torch.cat((neg_120, x1_120), dim=-1)
        neg_120 = x1_120 = None
        mul_545 = cat_121 * sin_63
        cat_121 = None
        q_embed_60 = mul_544 + mul_545
        mul_544 = mul_545 = None
        mul_546 = key_states_60 * cos_63
        cos_63 = None
        x1_121 = key_states_60[(Ellipsis, slice(None, 64, None))]
        x2_121 = key_states_60[(Ellipsis, slice(64, None, None))]
        key_states_60 = None
        neg_121 = -x2_121
        x2_121 = None
        cat_122 = torch.cat((neg_121, x1_121), dim=-1)
        neg_121 = x1_121 = None
        mul_547 = cat_122 * sin_63
        cat_122 = sin_63 = None
        k_embed_60 = mul_546 + mul_547
        mul_546 = mul_547 = None
        getitem_426 = k_embed_60[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_60 = None
        hidden_states_603 = getitem_426.expand(1, 8, 5, 19, 128)
        getitem_426 = None
        key_120 = hidden_states_603.reshape(1, 40, 19, 128)
        hidden_states_603 = None
        getitem_427 = value_states_60[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_60 = None
        hidden_states_604 = getitem_427.expand(1, 8, 5, 19, 128)
        getitem_427 = None
        value_120 = hidden_states_604.reshape(1, 40, 19, 128)
        hidden_states_604 = None
        attention_mask_61 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_60 = q_embed_60.contiguous()
        q_embed_60 = None
        key_121 = key_120.contiguous()
        key_120 = None
        value_121 = value_120.contiguous()
        value_120 = None
        attn_output_240 = torch._C._nn.scaled_dot_product_attention(
            query_60,
            key_121,
            value_121,
            attn_mask=attention_mask_61,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_60 = key_121 = value_121 = attention_mask_61 = None
        transpose_244 = attn_output_240.transpose(1, 2)
        attn_output_240 = None
        attn_output_241 = transpose_244.contiguous()
        transpose_244 = None
        reshape_182 = attn_output_241.reshape(1, 19, -1)
        attn_output_241 = None
        attn_output_242 = reshape_182.contiguous()
        reshape_182 = None
        attn_output_243 = torch._C._nn.linear(
            attn_output_242,
            l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_242 = l_self_modules_model_modules_layers_modules_60_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_605 = hidden_states_599 + attn_output_243
        hidden_states_599 = attn_output_243 = None
        hidden_states_606 = hidden_states_605.to(torch.float32)
        pow_122 = hidden_states_606.pow(2)
        variance_121 = pow_122.mean(-1, keepdim=True)
        pow_122 = None
        add_364 = variance_121 + 1e-05
        variance_121 = None
        rsqrt_121 = torch.rsqrt(add_364)
        add_364 = None
        hidden_states_607 = hidden_states_606 * rsqrt_121
        hidden_states_606 = rsqrt_121 = None
        to_247 = hidden_states_607.to(torch.float16)
        hidden_states_607 = None
        hidden_states_608 = (
            l_self_modules_model_modules_layers_modules_60_modules_post_attention_layernorm_parameters_weight_
            * to_247
        )
        l_self_modules_model_modules_layers_modules_60_modules_post_attention_layernorm_parameters_weight_ = (
            to_247
        ) = None
        linear_424 = torch._C._nn.linear(
            hidden_states_608,
            l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_60 = torch.nn.functional.silu(linear_424, inplace=False)
        linear_424 = None
        linear_425 = torch._C._nn.linear(
            hidden_states_608,
            l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_608 = l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_550 = silu_60 * linear_425
        silu_60 = linear_425 = None
        down_proj_60 = torch._C._nn.linear(
            mul_550,
            l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_550 = l_self_modules_model_modules_layers_modules_60_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_609 = hidden_states_605 + down_proj_60
        hidden_states_605 = down_proj_60 = None
        hidden_states_610 = hidden_states_609.to(torch.float32)
        pow_123 = hidden_states_610.pow(2)
        variance_122 = pow_123.mean(-1, keepdim=True)
        pow_123 = None
        add_366 = variance_122 + 1e-05
        variance_122 = None
        rsqrt_122 = torch.rsqrt(add_366)
        add_366 = None
        hidden_states_611 = hidden_states_610 * rsqrt_122
        hidden_states_610 = rsqrt_122 = None
        to_249 = hidden_states_611.to(torch.float16)
        hidden_states_611 = None
        hidden_states_612 = (
            l_self_modules_model_modules_layers_modules_61_modules_input_layernorm_parameters_weight_
            * to_249
        )
        l_self_modules_model_modules_layers_modules_61_modules_input_layernorm_parameters_weight_ = (
            to_249
        ) = None
        linear_427 = torch._C._nn.linear(
            hidden_states_612,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_183 = linear_427.view((1, 19, -1, 128))
        linear_427 = None
        query_states_61 = view_183.transpose(1, 2)
        view_183 = None
        linear_428 = torch._C._nn.linear(
            hidden_states_612,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_184 = linear_428.view((1, 19, -1, 128))
        linear_428 = None
        key_states_61 = view_184.transpose(1, 2)
        view_184 = None
        linear_429 = torch._C._nn.linear(
            hidden_states_612,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_612 = l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_185 = linear_429.view((1, 19, -1, 128))
        linear_429 = None
        value_states_61 = view_185.transpose(1, 2)
        view_185 = None
        cos_64 = cos_2.unsqueeze(1)
        sin_64 = sin_2.unsqueeze(1)
        mul_553 = query_states_61 * cos_64
        x1_122 = query_states_61[(Ellipsis, slice(None, 64, None))]
        x2_122 = query_states_61[(Ellipsis, slice(64, None, None))]
        query_states_61 = None
        neg_122 = -x2_122
        x2_122 = None
        cat_123 = torch.cat((neg_122, x1_122), dim=-1)
        neg_122 = x1_122 = None
        mul_554 = cat_123 * sin_64
        cat_123 = None
        q_embed_61 = mul_553 + mul_554
        mul_553 = mul_554 = None
        mul_555 = key_states_61 * cos_64
        cos_64 = None
        x1_123 = key_states_61[(Ellipsis, slice(None, 64, None))]
        x2_123 = key_states_61[(Ellipsis, slice(64, None, None))]
        key_states_61 = None
        neg_123 = -x2_123
        x2_123 = None
        cat_124 = torch.cat((neg_123, x1_123), dim=-1)
        neg_123 = x1_123 = None
        mul_556 = cat_124 * sin_64
        cat_124 = sin_64 = None
        k_embed_61 = mul_555 + mul_556
        mul_555 = mul_556 = None
        getitem_433 = k_embed_61[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_61 = None
        hidden_states_613 = getitem_433.expand(1, 8, 5, 19, 128)
        getitem_433 = None
        key_122 = hidden_states_613.reshape(1, 40, 19, 128)
        hidden_states_613 = None
        getitem_434 = value_states_61[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_61 = None
        hidden_states_614 = getitem_434.expand(1, 8, 5, 19, 128)
        getitem_434 = None
        value_122 = hidden_states_614.reshape(1, 40, 19, 128)
        hidden_states_614 = None
        attention_mask_62 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_61 = q_embed_61.contiguous()
        q_embed_61 = None
        key_123 = key_122.contiguous()
        key_122 = None
        value_123 = value_122.contiguous()
        value_122 = None
        attn_output_244 = torch._C._nn.scaled_dot_product_attention(
            query_61,
            key_123,
            value_123,
            attn_mask=attention_mask_62,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_61 = key_123 = value_123 = attention_mask_62 = None
        transpose_248 = attn_output_244.transpose(1, 2)
        attn_output_244 = None
        attn_output_245 = transpose_248.contiguous()
        transpose_248 = None
        reshape_185 = attn_output_245.reshape(1, 19, -1)
        attn_output_245 = None
        attn_output_246 = reshape_185.contiguous()
        reshape_185 = None
        attn_output_247 = torch._C._nn.linear(
            attn_output_246,
            l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_246 = l_self_modules_model_modules_layers_modules_61_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_615 = hidden_states_609 + attn_output_247
        hidden_states_609 = attn_output_247 = None
        hidden_states_616 = hidden_states_615.to(torch.float32)
        pow_124 = hidden_states_616.pow(2)
        variance_123 = pow_124.mean(-1, keepdim=True)
        pow_124 = None
        add_370 = variance_123 + 1e-05
        variance_123 = None
        rsqrt_123 = torch.rsqrt(add_370)
        add_370 = None
        hidden_states_617 = hidden_states_616 * rsqrt_123
        hidden_states_616 = rsqrt_123 = None
        to_251 = hidden_states_617.to(torch.float16)
        hidden_states_617 = None
        hidden_states_618 = (
            l_self_modules_model_modules_layers_modules_61_modules_post_attention_layernorm_parameters_weight_
            * to_251
        )
        l_self_modules_model_modules_layers_modules_61_modules_post_attention_layernorm_parameters_weight_ = (
            to_251
        ) = None
        linear_431 = torch._C._nn.linear(
            hidden_states_618,
            l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_61 = torch.nn.functional.silu(linear_431, inplace=False)
        linear_431 = None
        linear_432 = torch._C._nn.linear(
            hidden_states_618,
            l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_618 = l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_559 = silu_61 * linear_432
        silu_61 = linear_432 = None
        down_proj_61 = torch._C._nn.linear(
            mul_559,
            l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_559 = l_self_modules_model_modules_layers_modules_61_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_619 = hidden_states_615 + down_proj_61
        hidden_states_615 = down_proj_61 = None
        hidden_states_620 = hidden_states_619.to(torch.float32)
        pow_125 = hidden_states_620.pow(2)
        variance_124 = pow_125.mean(-1, keepdim=True)
        pow_125 = None
        add_372 = variance_124 + 1e-05
        variance_124 = None
        rsqrt_124 = torch.rsqrt(add_372)
        add_372 = None
        hidden_states_621 = hidden_states_620 * rsqrt_124
        hidden_states_620 = rsqrt_124 = None
        to_253 = hidden_states_621.to(torch.float16)
        hidden_states_621 = None
        hidden_states_622 = (
            l_self_modules_model_modules_layers_modules_62_modules_input_layernorm_parameters_weight_
            * to_253
        )
        l_self_modules_model_modules_layers_modules_62_modules_input_layernorm_parameters_weight_ = (
            to_253
        ) = None
        linear_434 = torch._C._nn.linear(
            hidden_states_622,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_186 = linear_434.view((1, 19, -1, 128))
        linear_434 = None
        query_states_62 = view_186.transpose(1, 2)
        view_186 = None
        linear_435 = torch._C._nn.linear(
            hidden_states_622,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_187 = linear_435.view((1, 19, -1, 128))
        linear_435 = None
        key_states_62 = view_187.transpose(1, 2)
        view_187 = None
        linear_436 = torch._C._nn.linear(
            hidden_states_622,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_622 = l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_188 = linear_436.view((1, 19, -1, 128))
        linear_436 = None
        value_states_62 = view_188.transpose(1, 2)
        view_188 = None
        cos_65 = cos_2.unsqueeze(1)
        sin_65 = sin_2.unsqueeze(1)
        mul_562 = query_states_62 * cos_65
        x1_124 = query_states_62[(Ellipsis, slice(None, 64, None))]
        x2_124 = query_states_62[(Ellipsis, slice(64, None, None))]
        query_states_62 = None
        neg_124 = -x2_124
        x2_124 = None
        cat_125 = torch.cat((neg_124, x1_124), dim=-1)
        neg_124 = x1_124 = None
        mul_563 = cat_125 * sin_65
        cat_125 = None
        q_embed_62 = mul_562 + mul_563
        mul_562 = mul_563 = None
        mul_564 = key_states_62 * cos_65
        cos_65 = None
        x1_125 = key_states_62[(Ellipsis, slice(None, 64, None))]
        x2_125 = key_states_62[(Ellipsis, slice(64, None, None))]
        key_states_62 = None
        neg_125 = -x2_125
        x2_125 = None
        cat_126 = torch.cat((neg_125, x1_125), dim=-1)
        neg_125 = x1_125 = None
        mul_565 = cat_126 * sin_65
        cat_126 = sin_65 = None
        k_embed_62 = mul_564 + mul_565
        mul_564 = mul_565 = None
        getitem_440 = k_embed_62[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_62 = None
        hidden_states_623 = getitem_440.expand(1, 8, 5, 19, 128)
        getitem_440 = None
        key_124 = hidden_states_623.reshape(1, 40, 19, 128)
        hidden_states_623 = None
        getitem_441 = value_states_62[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_62 = None
        hidden_states_624 = getitem_441.expand(1, 8, 5, 19, 128)
        getitem_441 = None
        value_124 = hidden_states_624.reshape(1, 40, 19, 128)
        hidden_states_624 = None
        attention_mask_63 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        query_62 = q_embed_62.contiguous()
        q_embed_62 = None
        key_125 = key_124.contiguous()
        key_124 = None
        value_125 = value_124.contiguous()
        value_124 = None
        attn_output_248 = torch._C._nn.scaled_dot_product_attention(
            query_62,
            key_125,
            value_125,
            attn_mask=attention_mask_63,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_62 = key_125 = value_125 = attention_mask_63 = None
        transpose_252 = attn_output_248.transpose(1, 2)
        attn_output_248 = None
        attn_output_249 = transpose_252.contiguous()
        transpose_252 = None
        reshape_188 = attn_output_249.reshape(1, 19, -1)
        attn_output_249 = None
        attn_output_250 = reshape_188.contiguous()
        reshape_188 = None
        attn_output_251 = torch._C._nn.linear(
            attn_output_250,
            l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_250 = l_self_modules_model_modules_layers_modules_62_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_625 = hidden_states_619 + attn_output_251
        hidden_states_619 = attn_output_251 = None
        hidden_states_626 = hidden_states_625.to(torch.float32)
        pow_126 = hidden_states_626.pow(2)
        variance_125 = pow_126.mean(-1, keepdim=True)
        pow_126 = None
        add_376 = variance_125 + 1e-05
        variance_125 = None
        rsqrt_125 = torch.rsqrt(add_376)
        add_376 = None
        hidden_states_627 = hidden_states_626 * rsqrt_125
        hidden_states_626 = rsqrt_125 = None
        to_255 = hidden_states_627.to(torch.float16)
        hidden_states_627 = None
        hidden_states_628 = (
            l_self_modules_model_modules_layers_modules_62_modules_post_attention_layernorm_parameters_weight_
            * to_255
        )
        l_self_modules_model_modules_layers_modules_62_modules_post_attention_layernorm_parameters_weight_ = (
            to_255
        ) = None
        linear_438 = torch._C._nn.linear(
            hidden_states_628,
            l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_62 = torch.nn.functional.silu(linear_438, inplace=False)
        linear_438 = None
        linear_439 = torch._C._nn.linear(
            hidden_states_628,
            l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_628 = l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_568 = silu_62 * linear_439
        silu_62 = linear_439 = None
        down_proj_62 = torch._C._nn.linear(
            mul_568,
            l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_568 = l_self_modules_model_modules_layers_modules_62_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_629 = hidden_states_625 + down_proj_62
        hidden_states_625 = down_proj_62 = None
        hidden_states_630 = hidden_states_629.to(torch.float32)
        pow_127 = hidden_states_630.pow(2)
        variance_126 = pow_127.mean(-1, keepdim=True)
        pow_127 = None
        add_378 = variance_126 + 1e-05
        variance_126 = None
        rsqrt_126 = torch.rsqrt(add_378)
        add_378 = None
        hidden_states_631 = hidden_states_630 * rsqrt_126
        hidden_states_630 = rsqrt_126 = None
        to_257 = hidden_states_631.to(torch.float16)
        hidden_states_631 = None
        hidden_states_632 = (
            l_self_modules_model_modules_layers_modules_63_modules_input_layernorm_parameters_weight_
            * to_257
        )
        l_self_modules_model_modules_layers_modules_63_modules_input_layernorm_parameters_weight_ = (
            to_257
        ) = None
        linear_441 = torch._C._nn.linear(
            hidden_states_632,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_q_proj_parameters_bias_ = (None)
        view_189 = linear_441.view((1, 19, -1, 128))
        linear_441 = None
        query_states_63 = view_189.transpose(1, 2)
        view_189 = None
        linear_442 = torch._C._nn.linear(
            hidden_states_632,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_bias_,
        )
        l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_k_proj_parameters_bias_ = (None)
        view_190 = linear_442.view((1, 19, -1, 128))
        linear_442 = None
        key_states_63 = view_190.transpose(1, 2)
        view_190 = None
        linear_443 = torch._C._nn.linear(
            hidden_states_632,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_weight_,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_bias_,
        )
        hidden_states_632 = l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_weight_ = l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_v_proj_parameters_bias_ = (None)
        view_191 = linear_443.view((1, 19, -1, 128))
        linear_443 = None
        value_states_63 = view_191.transpose(1, 2)
        view_191 = None
        cos_66 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_66 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_571 = query_states_63 * cos_66
        x1_126 = query_states_63[(Ellipsis, slice(None, 64, None))]
        x2_126 = query_states_63[(Ellipsis, slice(64, None, None))]
        query_states_63 = None
        neg_126 = -x2_126
        x2_126 = None
        cat_127 = torch.cat((neg_126, x1_126), dim=-1)
        neg_126 = x1_126 = None
        mul_572 = cat_127 * sin_66
        cat_127 = None
        q_embed_63 = mul_571 + mul_572
        mul_571 = mul_572 = None
        mul_573 = key_states_63 * cos_66
        cos_66 = None
        x1_127 = key_states_63[(Ellipsis, slice(None, 64, None))]
        x2_127 = key_states_63[(Ellipsis, slice(64, None, None))]
        key_states_63 = None
        neg_127 = -x2_127
        x2_127 = None
        cat_128 = torch.cat((neg_127, x1_127), dim=-1)
        neg_127 = x1_127 = None
        mul_574 = cat_128 * sin_66
        cat_128 = sin_66 = None
        k_embed_63 = mul_573 + mul_574
        mul_573 = mul_574 = None
        getitem_447 = k_embed_63[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_63 = None
        hidden_states_633 = getitem_447.expand(1, 8, 5, 19, 128)
        getitem_447 = None
        key_126 = hidden_states_633.reshape(1, 40, 19, 128)
        hidden_states_633 = None
        getitem_448 = value_states_63[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_63 = None
        hidden_states_634 = getitem_448.expand(1, 8, 5, 19, 128)
        getitem_448 = None
        value_126 = hidden_states_634.reshape(1, 40, 19, 128)
        hidden_states_634 = None
        attention_mask_64 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask = None
        query_63 = q_embed_63.contiguous()
        q_embed_63 = None
        key_127 = key_126.contiguous()
        key_126 = None
        value_127 = value_126.contiguous()
        value_126 = None
        attn_output_252 = torch._C._nn.scaled_dot_product_attention(
            query_63,
            key_127,
            value_127,
            attn_mask=attention_mask_64,
            dropout_p=0.0,
            scale=0.08838834764831845,
            is_causal=False,
        )
        query_63 = key_127 = value_127 = attention_mask_64 = None
        transpose_256 = attn_output_252.transpose(1, 2)
        attn_output_252 = None
        attn_output_253 = transpose_256.contiguous()
        transpose_256 = None
        reshape_191 = attn_output_253.reshape(1, 19, -1)
        attn_output_253 = None
        attn_output_254 = reshape_191.contiguous()
        reshape_191 = None
        attn_output_255 = torch._C._nn.linear(
            attn_output_254,
            l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_254 = l_self_modules_model_modules_layers_modules_63_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_635 = hidden_states_629 + attn_output_255
        hidden_states_629 = attn_output_255 = None
        hidden_states_636 = hidden_states_635.to(torch.float32)
        pow_128 = hidden_states_636.pow(2)
        variance_127 = pow_128.mean(-1, keepdim=True)
        pow_128 = None
        add_382 = variance_127 + 1e-05
        variance_127 = None
        rsqrt_127 = torch.rsqrt(add_382)
        add_382 = None
        hidden_states_637 = hidden_states_636 * rsqrt_127
        hidden_states_636 = rsqrt_127 = None
        to_259 = hidden_states_637.to(torch.float16)
        hidden_states_637 = None
        hidden_states_638 = (
            l_self_modules_model_modules_layers_modules_63_modules_post_attention_layernorm_parameters_weight_
            * to_259
        )
        l_self_modules_model_modules_layers_modules_63_modules_post_attention_layernorm_parameters_weight_ = (
            to_259
        ) = None
        linear_445 = torch._C._nn.linear(
            hidden_states_638,
            l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_63 = torch.nn.functional.silu(linear_445, inplace=False)
        linear_445 = None
        linear_446 = torch._C._nn.linear(
            hidden_states_638,
            l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_638 = l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_577 = silu_63 * linear_446
        silu_63 = linear_446 = None
        down_proj_63 = torch._C._nn.linear(
            mul_577,
            l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_577 = l_self_modules_model_modules_layers_modules_63_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_639 = hidden_states_635 + down_proj_63
        hidden_states_635 = down_proj_63 = None
        hidden_states_640 = hidden_states_639.to(torch.float32)
        hidden_states_639 = None
        pow_129 = hidden_states_640.pow(2)
        variance_128 = pow_129.mean(-1, keepdim=True)
        pow_129 = None
        add_384 = variance_128 + 1e-05
        variance_128 = None
        rsqrt_128 = torch.rsqrt(add_384)
        add_384 = None
        hidden_states_641 = hidden_states_640 * rsqrt_128
        hidden_states_640 = rsqrt_128 = None
        to_261 = hidden_states_641.to(torch.float16)
        hidden_states_641 = None
        hidden_states_642 = (
            l_self_modules_model_modules_norm_parameters_weight_ * to_261
        )
        l_self_modules_model_modules_norm_parameters_weight_ = to_261 = None
        getitem_450 = hidden_states_642[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_642 = None
        logits = torch._C._nn.linear(
            getitem_450, l_self_modules_lm_head_parameters_weight_, None
        )
        getitem_450 = l_self_modules_lm_head_parameters_weight_ = None
        return (logits,)
