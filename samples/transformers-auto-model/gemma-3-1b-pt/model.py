import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_embed_tokens_buffers_embed_scale_: torch.Tensor,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_input_ids_ = L_kwargs_input_ids_
        l_self_modules_model_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_embed_tokens_parameters_weight_
        )
        l_self_modules_model_modules_embed_tokens_buffers_embed_scale_ = (
            L_self_modules_model_modules_embed_tokens_buffers_embed_scale_
        )
        l_kwargs_attention_mask_ = L_kwargs_attention_mask_
        l_self_modules_model_modules_rotary_emb_buffers_inv_freq_ = (
            L_self_modules_model_modules_rotary_emb_buffers_inv_freq_
        )
        l_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_ = (
            L_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        embedding = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = None
        to = l_self_modules_model_modules_embed_tokens_buffers_embed_scale_.to(
            torch.float16
        )
        l_self_modules_model_modules_embed_tokens_buffers_embed_scale_ = None
        inputs_embeds = embedding * to
        embedding = to = None
        cache_position = torch.arange(0, 20, device=device(type="cpu"))
        position_ids = cache_position.unsqueeze(0)
        attention_mask = l_kwargs_attention_mask_.to(
            device=device(type="cpu"), dtype=torch.bool
        )
        kv_arange = torch.arange(20, device=device(type="cpu"))
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
            20, "error"
        )
        _vmap_increment_nesting_2 = None
        child_2 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        lazy_load_decompositions_3 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_3 = None
        _vmap_increment_nesting_3 = torch._C._functorch._vmap_increment_nesting(
            20, "error"
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
        batched_outputs = torch._C._functorch._remove_batch_dim(result_2, 4, 20, 0)
        result_2 = None
        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting = None
        batched_outputs_1 = torch._C._functorch._remove_batch_dim(
            batched_outputs, 3, 20, 0
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
            device=device(type="cpu"), dtype=torch.bool
        )
        l_kwargs_attention_mask_ = None
        kv_arange_2 = torch.arange(20, device=device(type="cpu"))
        kv_arange_2 += 0
        kv_arange_3 = kv_arange_2
        kv_arange_2 = None
        batch_arange_1 = torch.arange(1, device=device(type="cpu"))
        head_arange_1 = torch.arange(1, device=device(type="cpu"))
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
            20, "error"
        )
        _vmap_increment_nesting_6 = None
        child_6 = torch._C._functorch._add_batch_dim(cache_position, 0, 3)
        cache_position = None
        lazy_load_decompositions_7 = torch._functorch.vmap.lazy_load_decompositions()
        lazy_load_decompositions_7 = None
        _vmap_increment_nesting_7 = torch._C._functorch._vmap_increment_nesting(
            20, "error"
        )
        _vmap_increment_nesting_7 = None
        child_7 = torch._C._functorch._add_batch_dim(kv_arange_3, 0, 4)
        kv_arange_3 = None
        result_3 = child_6.new_ones((), dtype=torch.bool)
        result_4 = child_6.new_ones((), dtype=torch.bool)
        sub = child_6.sub(512)
        gt = child_7.gt(sub)
        sub = None
        result_5 = result_4.__and__(gt)
        result_4 = gt = None
        le_1 = child_7.le(child_6)
        child_6 = None
        result_6 = result_5.__and__(le_1)
        result_5 = le_1 = None
        result_7 = result_3.__and__(result_6)
        result_3 = result_6 = None
        function_ctx_1 = torch.autograd.function.FunctionCtx()
        function_ctx_1 = None
        index_1 = torch.ops.aten.index(attention_mask_1, [child_4, child_7])
        attention_mask_1 = child_4 = child_7 = None
        result_8 = result_7.__and__(index_1)
        result_7 = index_1 = None
        batched_outputs_3 = torch._C._functorch._remove_batch_dim(result_8, 4, 20, 0)
        result_8 = None
        _vmap_decrement_nesting_4 = torch._C._functorch._vmap_decrement_nesting()
        _vmap_decrement_nesting_4 = None
        batched_outputs_4 = torch._C._functorch._remove_batch_dim(
            batched_outputs_3, 3, 20, 0
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
        cos_10 = cos_1.to(dtype=torch.float16)
        cos_1 = None
        sin_10 = sin_1.to(dtype=torch.float16)
        sin_1 = None
        getitem_2 = l_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_[
            (None, slice(None, None, None), None)
        ]
        l_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_ = None
        float_5 = getitem_2.float()
        getitem_2 = None
        expand_1 = float_5.expand(1, -1, 1)
        float_5 = None
        inv_freq_expanded_1 = expand_1.to(device(type="cpu"))
        expand_1 = None
        getitem_3 = position_ids[
            (slice(None, None, None), None, slice(None, None, None))
        ]
        position_ids = None
        position_ids_expanded_1 = getitem_3.float()
        getitem_3 = None
        _enter_autocast_1 = torch.amp.autocast_mode._enter_autocast(
            "cpu", None, False, None
        )
        float_7 = inv_freq_expanded_1.float()
        inv_freq_expanded_1 = None
        float_8 = position_ids_expanded_1.float()
        position_ids_expanded_1 = None
        matmul_1 = float_7 @ float_8
        float_7 = float_8 = None
        freqs_1 = matmul_1.transpose(1, 2)
        matmul_1 = None
        emb_1 = torch.cat((freqs_1, freqs_1), dim=-1)
        freqs_1 = None
        cos_2 = emb_1.cos()
        cos_3 = cos_2 * 1.0
        cos_2 = None
        sin_2 = emb_1.sin()
        emb_1 = None
        sin_3 = sin_2 * 1.0
        sin_2 = None
        _exit_autocast_1 = torch.amp.autocast_mode._exit_autocast(_enter_autocast_1)
        _enter_autocast_1 = _exit_autocast_1 = None
        cos_4 = cos_3.to(dtype=torch.float16)
        cos_3 = None
        sin_4 = sin_3.to(dtype=torch.float16)
        sin_3 = None
        _log_api_usage_once = torch._C._log_api_usage_once("python.nn_module")
        _log_api_usage_once = None
        float_9 = inputs_embeds.float()
        pow_1 = float_9.pow(2)
        mean = pow_1.mean(-1, keepdim=True)
        pow_1 = None
        add = mean + 1e-06
        mean = None
        rsqrt = torch.rsqrt(add)
        add = None
        output = float_9 * rsqrt
        float_9 = rsqrt = None
        float_10 = (
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_1 = 1.0 + float_10
        float_10 = None
        output_1 = output * add_1
        output = add_1 = None
        hidden_states = output_1.type_as(inputs_embeds)
        output_1 = None
        linear = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 20, -1, 256))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 20, -1, 256))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = linear_2.view((1, 20, -1, 256))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        float_11 = query_states.float()
        pow_2 = float_11.pow(2)
        mean_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_2 = mean_1 + 1e-06
        mean_1 = None
        rsqrt_1 = torch.rsqrt(add_2)
        add_2 = None
        output_2 = float_11 * rsqrt_1
        float_11 = rsqrt_1 = None
        float_12 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_3 = 1.0 + float_12
        float_12 = None
        output_3 = output_2 * add_3
        output_2 = add_3 = None
        query_states_1 = output_3.type_as(query_states)
        output_3 = query_states = None
        float_13 = key_states.float()
        pow_3 = float_13.pow(2)
        mean_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        add_4 = mean_2 + 1e-06
        mean_2 = None
        rsqrt_2 = torch.rsqrt(add_4)
        add_4 = None
        output_4 = float_13 * rsqrt_2
        float_13 = rsqrt_2 = None
        float_14 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_5 = 1.0 + float_14
        float_14 = None
        output_5 = output_4 * add_5
        output_4 = add_5 = None
        key_states_1 = output_5.type_as(key_states)
        output_5 = key_states = None
        cos_5 = cos_4.unsqueeze(1)
        sin_5 = sin_4.unsqueeze(1)
        mul_11 = query_states_1 * cos_5
        x1 = query_states_1[(Ellipsis, slice(None, 128, None))]
        x2 = query_states_1[(Ellipsis, slice(128, None, None))]
        query_states_1 = None
        neg = -x2
        x2 = None
        cat_2 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_12 = cat_2 * sin_5
        cat_2 = None
        q_embed = mul_11 + mul_12
        mul_11 = mul_12 = None
        mul_13 = key_states_1 * cos_5
        cos_5 = None
        x1_1 = key_states_1[(Ellipsis, slice(None, 128, None))]
        x2_1 = key_states_1[(Ellipsis, slice(128, None, None))]
        key_states_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_3 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_14 = cat_3 * sin_5
        cat_3 = sin_5 = None
        k_embed = mul_13 + mul_14
        mul_13 = mul_14 = None
        getitem_8 = k_embed[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed = None
        hidden_states_1 = getitem_8.expand(1, 1, 4, 20, 256)
        getitem_8 = None
        key = hidden_states_1.reshape(1, 4, 20, 256)
        hidden_states_1 = None
        getitem_9 = value_states[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states = None
        hidden_states_2 = getitem_9.expand(1, 1, 4, 20, 256)
        getitem_9 = None
        value = hidden_states_2.reshape(1, 4, 20, 256)
        hidden_states_2 = None
        attention_mask_2 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query = key_1 = value_1 = attention_mask_2 = None
        transpose_5 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_5.contiguous()
        transpose_5 = None
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
        float_15 = attn_output_3.float()
        pow_4 = float_15.pow(2)
        mean_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_8 = mean_3 + 1e-06
        mean_3 = None
        rsqrt_3 = torch.rsqrt(add_8)
        add_8 = None
        output_6 = float_15 * rsqrt_3
        float_15 = rsqrt_3 = None
        float_16 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_9 = 1.0 + float_16
        float_16 = None
        output_7 = output_6 * add_9
        output_6 = add_9 = None
        hidden_states_3 = output_7.type_as(attn_output_3)
        output_7 = attn_output_3 = None
        hidden_states_4 = inputs_embeds + hidden_states_3
        inputs_embeds = hidden_states_3 = None
        float_17 = hidden_states_4.float()
        pow_5 = float_17.pow(2)
        mean_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_11 = mean_4 + 1e-06
        mean_4 = None
        rsqrt_4 = torch.rsqrt(add_11)
        add_11 = None
        output_8 = float_17 * rsqrt_4
        float_17 = rsqrt_4 = None
        float_18 = (
            l_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_12 = 1.0 + float_18
        float_18 = None
        output_9 = output_8 * add_12
        output_8 = add_12 = None
        hidden_states_5 = output_9.type_as(hidden_states_4)
        output_9 = None
        linear_4 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu = torch._C._nn.gelu(linear_4, approximate="tanh")
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_5 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_19 = gelu * linear_5
        gelu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_19,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_19 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_19 = down_proj.float()
        pow_6 = float_19.pow(2)
        mean_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_13 = mean_5 + 1e-06
        mean_5 = None
        rsqrt_5 = torch.rsqrt(add_13)
        add_13 = None
        output_10 = float_19 * rsqrt_5
        float_19 = rsqrt_5 = None
        float_20 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_14 = 1.0 + float_20
        float_20 = None
        output_11 = output_10 * add_14
        output_10 = add_14 = None
        hidden_states_6 = output_11.type_as(down_proj)
        output_11 = down_proj = None
        hidden_states_7 = hidden_states_4 + hidden_states_6
        hidden_states_4 = hidden_states_6 = None
        float_21 = hidden_states_7.float()
        pow_7 = float_21.pow(2)
        mean_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        add_16 = mean_6 + 1e-06
        mean_6 = None
        rsqrt_6 = torch.rsqrt(add_16)
        add_16 = None
        output_12 = float_21 * rsqrt_6
        float_21 = rsqrt_6 = None
        float_22 = (
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_17 = 1.0 + float_22
        float_22 = None
        output_13 = output_12 * add_17
        output_12 = add_17 = None
        hidden_states_8 = output_13.type_as(hidden_states_7)
        output_13 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 20, -1, 256))
        linear_7 = None
        query_states_2 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 20, -1, 256))
        linear_8 = None
        key_states_2 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_8,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_8 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = linear_9.view((1, 20, -1, 256))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        float_23 = query_states_2.float()
        pow_8 = float_23.pow(2)
        mean_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_18 = mean_7 + 1e-06
        mean_7 = None
        rsqrt_7 = torch.rsqrt(add_18)
        add_18 = None
        output_14 = float_23 * rsqrt_7
        float_23 = rsqrt_7 = None
        float_24 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_19 = 1.0 + float_24
        float_24 = None
        output_15 = output_14 * add_19
        output_14 = add_19 = None
        query_states_3 = output_15.type_as(query_states_2)
        output_15 = query_states_2 = None
        float_25 = key_states_2.float()
        pow_9 = float_25.pow(2)
        mean_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_20 = mean_8 + 1e-06
        mean_8 = None
        rsqrt_8 = torch.rsqrt(add_20)
        add_20 = None
        output_16 = float_25 * rsqrt_8
        float_25 = rsqrt_8 = None
        float_26 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_21 = 1.0 + float_26
        float_26 = None
        output_17 = output_16 * add_21
        output_16 = add_21 = None
        key_states_3 = output_17.type_as(key_states_2)
        output_17 = key_states_2 = None
        cos_6 = cos_4.unsqueeze(1)
        sin_6 = sin_4.unsqueeze(1)
        mul_28 = query_states_3 * cos_6
        x1_2 = query_states_3[(Ellipsis, slice(None, 128, None))]
        x2_2 = query_states_3[(Ellipsis, slice(128, None, None))]
        query_states_3 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_4 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_29 = cat_4 * sin_6
        cat_4 = None
        q_embed_1 = mul_28 + mul_29
        mul_28 = mul_29 = None
        mul_30 = key_states_3 * cos_6
        cos_6 = None
        x1_3 = key_states_3[(Ellipsis, slice(None, 128, None))]
        x2_3 = key_states_3[(Ellipsis, slice(128, None, None))]
        key_states_3 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_5 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_31 = cat_5 * sin_6
        cat_5 = sin_6 = None
        k_embed_1 = mul_30 + mul_31
        mul_30 = mul_31 = None
        getitem_15 = k_embed_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_1 = None
        hidden_states_9 = getitem_15.expand(1, 1, 4, 20, 256)
        getitem_15 = None
        key_2 = hidden_states_9.reshape(1, 4, 20, 256)
        hidden_states_9 = None
        getitem_16 = value_states_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_1 = None
        hidden_states_10 = getitem_16.expand(1, 1, 4, 20, 256)
        getitem_16 = None
        value_2 = hidden_states_10.reshape(1, 4, 20, 256)
        hidden_states_10 = None
        attention_mask_3 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_1 = key_3 = value_3 = attention_mask_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
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
        float_27 = attn_output_7.float()
        pow_10 = float_27.pow(2)
        mean_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_24 = mean_9 + 1e-06
        mean_9 = None
        rsqrt_9 = torch.rsqrt(add_24)
        add_24 = None
        output_18 = float_27 * rsqrt_9
        float_27 = rsqrt_9 = None
        float_28 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_25 = 1.0 + float_28
        float_28 = None
        output_19 = output_18 * add_25
        output_18 = add_25 = None
        hidden_states_11 = output_19.type_as(attn_output_7)
        output_19 = attn_output_7 = None
        hidden_states_12 = hidden_states_7 + hidden_states_11
        hidden_states_7 = hidden_states_11 = None
        float_29 = hidden_states_12.float()
        pow_11 = float_29.pow(2)
        mean_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        add_27 = mean_10 + 1e-06
        mean_10 = None
        rsqrt_10 = torch.rsqrt(add_27)
        add_27 = None
        output_20 = float_29 * rsqrt_10
        float_29 = rsqrt_10 = None
        float_30 = (
            l_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_28 = 1.0 + float_30
        float_30 = None
        output_21 = output_20 * add_28
        output_20 = add_28 = None
        hidden_states_13 = output_21.type_as(hidden_states_12)
        output_21 = None
        linear_11 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_1 = torch._C._nn.gelu(linear_11, approximate="tanh")
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_13 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_36 = gelu_1 * linear_12
        gelu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_36,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_36 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_31 = down_proj_1.float()
        pow_12 = float_31.pow(2)
        mean_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_29 = mean_11 + 1e-06
        mean_11 = None
        rsqrt_11 = torch.rsqrt(add_29)
        add_29 = None
        output_22 = float_31 * rsqrt_11
        float_31 = rsqrt_11 = None
        float_32 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_30 = 1.0 + float_32
        float_32 = None
        output_23 = output_22 * add_30
        output_22 = add_30 = None
        hidden_states_14 = output_23.type_as(down_proj_1)
        output_23 = down_proj_1 = None
        hidden_states_15 = hidden_states_12 + hidden_states_14
        hidden_states_12 = hidden_states_14 = None
        float_33 = hidden_states_15.float()
        pow_13 = float_33.pow(2)
        mean_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_32 = mean_12 + 1e-06
        mean_12 = None
        rsqrt_12 = torch.rsqrt(add_32)
        add_32 = None
        output_24 = float_33 * rsqrt_12
        float_33 = rsqrt_12 = None
        float_34 = (
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_33 = 1.0 + float_34
        float_34 = None
        output_25 = output_24 * add_33
        output_24 = add_33 = None
        hidden_states_16 = output_25.type_as(hidden_states_15)
        output_25 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 20, -1, 256))
        linear_14 = None
        query_states_4 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 20, -1, 256))
        linear_15 = None
        key_states_4 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_16,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_16 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = linear_16.view((1, 20, -1, 256))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        float_35 = query_states_4.float()
        pow_14 = float_35.pow(2)
        mean_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_34 = mean_13 + 1e-06
        mean_13 = None
        rsqrt_13 = torch.rsqrt(add_34)
        add_34 = None
        output_26 = float_35 * rsqrt_13
        float_35 = rsqrt_13 = None
        float_36 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_35 = 1.0 + float_36
        float_36 = None
        output_27 = output_26 * add_35
        output_26 = add_35 = None
        query_states_5 = output_27.type_as(query_states_4)
        output_27 = query_states_4 = None
        float_37 = key_states_4.float()
        pow_15 = float_37.pow(2)
        mean_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        add_36 = mean_14 + 1e-06
        mean_14 = None
        rsqrt_14 = torch.rsqrt(add_36)
        add_36 = None
        output_28 = float_37 * rsqrt_14
        float_37 = rsqrt_14 = None
        float_38 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_37 = 1.0 + float_38
        float_38 = None
        output_29 = output_28 * add_37
        output_28 = add_37 = None
        key_states_5 = output_29.type_as(key_states_4)
        output_29 = key_states_4 = None
        cos_7 = cos_4.unsqueeze(1)
        sin_7 = sin_4.unsqueeze(1)
        mul_45 = query_states_5 * cos_7
        x1_4 = query_states_5[(Ellipsis, slice(None, 128, None))]
        x2_4 = query_states_5[(Ellipsis, slice(128, None, None))]
        query_states_5 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_6 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_46 = cat_6 * sin_7
        cat_6 = None
        q_embed_2 = mul_45 + mul_46
        mul_45 = mul_46 = None
        mul_47 = key_states_5 * cos_7
        cos_7 = None
        x1_5 = key_states_5[(Ellipsis, slice(None, 128, None))]
        x2_5 = key_states_5[(Ellipsis, slice(128, None, None))]
        key_states_5 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_7 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_48 = cat_7 * sin_7
        cat_7 = sin_7 = None
        k_embed_2 = mul_47 + mul_48
        mul_47 = mul_48 = None
        getitem_22 = k_embed_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_2 = None
        hidden_states_17 = getitem_22.expand(1, 1, 4, 20, 256)
        getitem_22 = None
        key_4 = hidden_states_17.reshape(1, 4, 20, 256)
        hidden_states_17 = None
        getitem_23 = value_states_2[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_2 = None
        hidden_states_18 = getitem_23.expand(1, 1, 4, 20, 256)
        getitem_23 = None
        value_4 = hidden_states_18.reshape(1, 4, 20, 256)
        hidden_states_18 = None
        attention_mask_4 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_2 = key_5 = value_5 = attention_mask_4 = None
        transpose_13 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_13.contiguous()
        transpose_13 = None
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
        float_39 = attn_output_11.float()
        pow_16 = float_39.pow(2)
        mean_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_40 = mean_15 + 1e-06
        mean_15 = None
        rsqrt_15 = torch.rsqrt(add_40)
        add_40 = None
        output_30 = float_39 * rsqrt_15
        float_39 = rsqrt_15 = None
        float_40 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_41 = 1.0 + float_40
        float_40 = None
        output_31 = output_30 * add_41
        output_30 = add_41 = None
        hidden_states_19 = output_31.type_as(attn_output_11)
        output_31 = attn_output_11 = None
        hidden_states_20 = hidden_states_15 + hidden_states_19
        hidden_states_15 = hidden_states_19 = None
        float_41 = hidden_states_20.float()
        pow_17 = float_41.pow(2)
        mean_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_43 = mean_16 + 1e-06
        mean_16 = None
        rsqrt_16 = torch.rsqrt(add_43)
        add_43 = None
        output_32 = float_41 * rsqrt_16
        float_41 = rsqrt_16 = None
        float_42 = (
            l_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_44 = 1.0 + float_42
        float_42 = None
        output_33 = output_32 * add_44
        output_32 = add_44 = None
        hidden_states_21 = output_33.type_as(hidden_states_20)
        output_33 = None
        linear_18 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_2 = torch._C._nn.gelu(linear_18, approximate="tanh")
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_21,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_21 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_53 = gelu_2 * linear_19
        gelu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_53,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_53 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_43 = down_proj_2.float()
        pow_18 = float_43.pow(2)
        mean_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_45 = mean_17 + 1e-06
        mean_17 = None
        rsqrt_17 = torch.rsqrt(add_45)
        add_45 = None
        output_34 = float_43 * rsqrt_17
        float_43 = rsqrt_17 = None
        float_44 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_46 = 1.0 + float_44
        float_44 = None
        output_35 = output_34 * add_46
        output_34 = add_46 = None
        hidden_states_22 = output_35.type_as(down_proj_2)
        output_35 = down_proj_2 = None
        hidden_states_23 = hidden_states_20 + hidden_states_22
        hidden_states_20 = hidden_states_22 = None
        float_45 = hidden_states_23.float()
        pow_19 = float_45.pow(2)
        mean_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        add_48 = mean_18 + 1e-06
        mean_18 = None
        rsqrt_18 = torch.rsqrt(add_48)
        add_48 = None
        output_36 = float_45 * rsqrt_18
        float_45 = rsqrt_18 = None
        float_46 = (
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_49 = 1.0 + float_46
        float_46 = None
        output_37 = output_36 * add_49
        output_36 = add_49 = None
        hidden_states_24 = output_37.type_as(hidden_states_23)
        output_37 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 20, -1, 256))
        linear_21 = None
        query_states_6 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 20, -1, 256))
        linear_22 = None
        key_states_6 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_24,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_24 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = linear_23.view((1, 20, -1, 256))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        float_47 = query_states_6.float()
        pow_20 = float_47.pow(2)
        mean_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_50 = mean_19 + 1e-06
        mean_19 = None
        rsqrt_19 = torch.rsqrt(add_50)
        add_50 = None
        output_38 = float_47 * rsqrt_19
        float_47 = rsqrt_19 = None
        float_48 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_51 = 1.0 + float_48
        float_48 = None
        output_39 = output_38 * add_51
        output_38 = add_51 = None
        query_states_7 = output_39.type_as(query_states_6)
        output_39 = query_states_6 = None
        float_49 = key_states_6.float()
        pow_21 = float_49.pow(2)
        mean_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_52 = mean_20 + 1e-06
        mean_20 = None
        rsqrt_20 = torch.rsqrt(add_52)
        add_52 = None
        output_40 = float_49 * rsqrt_20
        float_49 = rsqrt_20 = None
        float_50 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_53 = 1.0 + float_50
        float_50 = None
        output_41 = output_40 * add_53
        output_40 = add_53 = None
        key_states_7 = output_41.type_as(key_states_6)
        output_41 = key_states_6 = None
        cos_8 = cos_4.unsqueeze(1)
        sin_8 = sin_4.unsqueeze(1)
        mul_62 = query_states_7 * cos_8
        x1_6 = query_states_7[(Ellipsis, slice(None, 128, None))]
        x2_6 = query_states_7[(Ellipsis, slice(128, None, None))]
        query_states_7 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_8 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_63 = cat_8 * sin_8
        cat_8 = None
        q_embed_3 = mul_62 + mul_63
        mul_62 = mul_63 = None
        mul_64 = key_states_7 * cos_8
        cos_8 = None
        x1_7 = key_states_7[(Ellipsis, slice(None, 128, None))]
        x2_7 = key_states_7[(Ellipsis, slice(128, None, None))]
        key_states_7 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_9 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_65 = cat_9 * sin_8
        cat_9 = sin_8 = None
        k_embed_3 = mul_64 + mul_65
        mul_64 = mul_65 = None
        getitem_29 = k_embed_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_3 = None
        hidden_states_25 = getitem_29.expand(1, 1, 4, 20, 256)
        getitem_29 = None
        key_6 = hidden_states_25.reshape(1, 4, 20, 256)
        hidden_states_25 = None
        getitem_30 = value_states_3[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_3 = None
        hidden_states_26 = getitem_30.expand(1, 1, 4, 20, 256)
        getitem_30 = None
        value_6 = hidden_states_26.reshape(1, 4, 20, 256)
        hidden_states_26 = None
        attention_mask_5 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_3 = key_7 = value_7 = attention_mask_5 = None
        transpose_17 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_17.contiguous()
        transpose_17 = None
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
        float_51 = attn_output_15.float()
        pow_22 = float_51.pow(2)
        mean_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_56 = mean_21 + 1e-06
        mean_21 = None
        rsqrt_21 = torch.rsqrt(add_56)
        add_56 = None
        output_42 = float_51 * rsqrt_21
        float_51 = rsqrt_21 = None
        float_52 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_57 = 1.0 + float_52
        float_52 = None
        output_43 = output_42 * add_57
        output_42 = add_57 = None
        hidden_states_27 = output_43.type_as(attn_output_15)
        output_43 = attn_output_15 = None
        hidden_states_28 = hidden_states_23 + hidden_states_27
        hidden_states_23 = hidden_states_27 = None
        float_53 = hidden_states_28.float()
        pow_23 = float_53.pow(2)
        mean_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        add_59 = mean_22 + 1e-06
        mean_22 = None
        rsqrt_22 = torch.rsqrt(add_59)
        add_59 = None
        output_44 = float_53 * rsqrt_22
        float_53 = rsqrt_22 = None
        float_54 = (
            l_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_60 = 1.0 + float_54
        float_54 = None
        output_45 = output_44 * add_60
        output_44 = add_60 = None
        hidden_states_29 = output_45.type_as(hidden_states_28)
        output_45 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_3 = torch._C._nn.gelu(linear_25, approximate="tanh")
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_29,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_29 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_70 = gelu_3 * linear_26
        gelu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_70,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_70 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_55 = down_proj_3.float()
        pow_24 = float_55.pow(2)
        mean_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_61 = mean_23 + 1e-06
        mean_23 = None
        rsqrt_23 = torch.rsqrt(add_61)
        add_61 = None
        output_46 = float_55 * rsqrt_23
        float_55 = rsqrt_23 = None
        float_56 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_62 = 1.0 + float_56
        float_56 = None
        output_47 = output_46 * add_62
        output_46 = add_62 = None
        hidden_states_30 = output_47.type_as(down_proj_3)
        output_47 = down_proj_3 = None
        hidden_states_31 = hidden_states_28 + hidden_states_30
        hidden_states_28 = hidden_states_30 = None
        float_57 = hidden_states_31.float()
        pow_25 = float_57.pow(2)
        mean_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_64 = mean_24 + 1e-06
        mean_24 = None
        rsqrt_24 = torch.rsqrt(add_64)
        add_64 = None
        output_48 = float_57 * rsqrt_24
        float_57 = rsqrt_24 = None
        float_58 = (
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_65 = 1.0 + float_58
        float_58 = None
        output_49 = output_48 * add_65
        output_48 = add_65 = None
        hidden_states_32 = output_49.type_as(hidden_states_31)
        output_49 = None
        linear_28 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 20, -1, 256))
        linear_28 = None
        query_states_8 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 20, -1, 256))
        linear_29 = None
        key_states_8 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = linear_30.view((1, 20, -1, 256))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        float_59 = query_states_8.float()
        pow_26 = float_59.pow(2)
        mean_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_66 = mean_25 + 1e-06
        mean_25 = None
        rsqrt_25 = torch.rsqrt(add_66)
        add_66 = None
        output_50 = float_59 * rsqrt_25
        float_59 = rsqrt_25 = None
        float_60 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_67 = 1.0 + float_60
        float_60 = None
        output_51 = output_50 * add_67
        output_50 = add_67 = None
        query_states_9 = output_51.type_as(query_states_8)
        output_51 = query_states_8 = None
        float_61 = key_states_8.float()
        pow_27 = float_61.pow(2)
        mean_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        add_68 = mean_26 + 1e-06
        mean_26 = None
        rsqrt_26 = torch.rsqrt(add_68)
        add_68 = None
        output_52 = float_61 * rsqrt_26
        float_61 = rsqrt_26 = None
        float_62 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_69 = 1.0 + float_62
        float_62 = None
        output_53 = output_52 * add_69
        output_52 = add_69 = None
        key_states_9 = output_53.type_as(key_states_8)
        output_53 = key_states_8 = None
        cos_9 = cos_4.unsqueeze(1)
        sin_9 = sin_4.unsqueeze(1)
        mul_79 = query_states_9 * cos_9
        x1_8 = query_states_9[(Ellipsis, slice(None, 128, None))]
        x2_8 = query_states_9[(Ellipsis, slice(128, None, None))]
        query_states_9 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_10 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_80 = cat_10 * sin_9
        cat_10 = None
        q_embed_4 = mul_79 + mul_80
        mul_79 = mul_80 = None
        mul_81 = key_states_9 * cos_9
        cos_9 = None
        x1_9 = key_states_9[(Ellipsis, slice(None, 128, None))]
        x2_9 = key_states_9[(Ellipsis, slice(128, None, None))]
        key_states_9 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_11 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_82 = cat_11 * sin_9
        cat_11 = sin_9 = None
        k_embed_4 = mul_81 + mul_82
        mul_81 = mul_82 = None
        getitem_36 = k_embed_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_4 = None
        hidden_states_33 = getitem_36.expand(1, 1, 4, 20, 256)
        getitem_36 = None
        key_8 = hidden_states_33.reshape(1, 4, 20, 256)
        hidden_states_33 = None
        getitem_37 = value_states_4[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_4 = None
        hidden_states_34 = getitem_37.expand(1, 1, 4, 20, 256)
        getitem_37 = None
        value_8 = hidden_states_34.reshape(1, 4, 20, 256)
        hidden_states_34 = None
        attention_mask_6 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_4 = key_9 = value_9 = attention_mask_6 = None
        transpose_21 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_21.contiguous()
        transpose_21 = None
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
        float_63 = attn_output_19.float()
        pow_28 = float_63.pow(2)
        mean_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_72 = mean_27 + 1e-06
        mean_27 = None
        rsqrt_27 = torch.rsqrt(add_72)
        add_72 = None
        output_54 = float_63 * rsqrt_27
        float_63 = rsqrt_27 = None
        float_64 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_73 = 1.0 + float_64
        float_64 = None
        output_55 = output_54 * add_73
        output_54 = add_73 = None
        hidden_states_35 = output_55.type_as(attn_output_19)
        output_55 = attn_output_19 = None
        hidden_states_36 = hidden_states_31 + hidden_states_35
        hidden_states_31 = hidden_states_35 = None
        float_65 = hidden_states_36.float()
        pow_29 = float_65.pow(2)
        mean_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_75 = mean_28 + 1e-06
        mean_28 = None
        rsqrt_28 = torch.rsqrt(add_75)
        add_75 = None
        output_56 = float_65 * rsqrt_28
        float_65 = rsqrt_28 = None
        float_66 = (
            l_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_76 = 1.0 + float_66
        float_66 = None
        output_57 = output_56 * add_76
        output_56 = add_76 = None
        hidden_states_37 = output_57.type_as(hidden_states_36)
        output_57 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_4 = torch._C._nn.gelu(linear_32, approximate="tanh")
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_37 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_87 = gelu_4 * linear_33
        gelu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_87,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_87 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_67 = down_proj_4.float()
        pow_30 = float_67.pow(2)
        mean_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_77 = mean_29 + 1e-06
        mean_29 = None
        rsqrt_29 = torch.rsqrt(add_77)
        add_77 = None
        output_58 = float_67 * rsqrt_29
        float_67 = rsqrt_29 = None
        float_68 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_78 = 1.0 + float_68
        float_68 = None
        output_59 = output_58 * add_78
        output_58 = add_78 = None
        hidden_states_38 = output_59.type_as(down_proj_4)
        output_59 = down_proj_4 = None
        hidden_states_39 = hidden_states_36 + hidden_states_38
        hidden_states_36 = hidden_states_38 = None
        float_69 = hidden_states_39.float()
        pow_31 = float_69.pow(2)
        mean_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        add_80 = mean_30 + 1e-06
        mean_30 = None
        rsqrt_30 = torch.rsqrt(add_80)
        add_80 = None
        output_60 = float_69 * rsqrt_30
        float_69 = rsqrt_30 = None
        float_70 = (
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_81 = 1.0 + float_70
        float_70 = None
        output_61 = output_60 * add_81
        output_60 = add_81 = None
        hidden_states_40 = output_61.type_as(hidden_states_39)
        output_61 = None
        linear_35 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 20, -1, 256))
        linear_35 = None
        query_states_10 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 20, -1, 256))
        linear_36 = None
        key_states_10 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_40 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = linear_37.view((1, 20, -1, 256))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        float_71 = query_states_10.float()
        pow_32 = float_71.pow(2)
        mean_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_82 = mean_31 + 1e-06
        mean_31 = None
        rsqrt_31 = torch.rsqrt(add_82)
        add_82 = None
        output_62 = float_71 * rsqrt_31
        float_71 = rsqrt_31 = None
        float_72 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_83 = 1.0 + float_72
        float_72 = None
        output_63 = output_62 * add_83
        output_62 = add_83 = None
        query_states_11 = output_63.type_as(query_states_10)
        output_63 = query_states_10 = None
        float_73 = key_states_10.float()
        pow_33 = float_73.pow(2)
        mean_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_84 = mean_32 + 1e-06
        mean_32 = None
        rsqrt_32 = torch.rsqrt(add_84)
        add_84 = None
        output_64 = float_73 * rsqrt_32
        float_73 = rsqrt_32 = None
        float_74 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_85 = 1.0 + float_74
        float_74 = None
        output_65 = output_64 * add_85
        output_64 = add_85 = None
        key_states_11 = output_65.type_as(key_states_10)
        output_65 = key_states_10 = None
        cos_11 = cos_10.unsqueeze(1)
        sin_11 = sin_10.unsqueeze(1)
        mul_96 = query_states_11 * cos_11
        x1_10 = query_states_11[(Ellipsis, slice(None, 128, None))]
        x2_10 = query_states_11[(Ellipsis, slice(128, None, None))]
        query_states_11 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_12 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_97 = cat_12 * sin_11
        cat_12 = None
        q_embed_5 = mul_96 + mul_97
        mul_96 = mul_97 = None
        mul_98 = key_states_11 * cos_11
        cos_11 = None
        x1_11 = key_states_11[(Ellipsis, slice(None, 128, None))]
        x2_11 = key_states_11[(Ellipsis, slice(128, None, None))]
        key_states_11 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_13 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_99 = cat_13 * sin_11
        cat_13 = sin_11 = None
        k_embed_5 = mul_98 + mul_99
        mul_98 = mul_99 = None
        getitem_43 = k_embed_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_5 = None
        hidden_states_41 = getitem_43.expand(1, 1, 4, 20, 256)
        getitem_43 = None
        key_10 = hidden_states_41.reshape(1, 4, 20, 256)
        hidden_states_41 = None
        getitem_44 = value_states_5[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_5 = None
        hidden_states_42 = getitem_44.expand(1, 1, 4, 20, 256)
        getitem_44 = None
        value_10 = hidden_states_42.reshape(1, 4, 20, 256)
        hidden_states_42 = None
        attention_mask_7 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_5 = key_11 = value_11 = attention_mask_7 = None
        transpose_25 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_25.contiguous()
        transpose_25 = None
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
        float_75 = attn_output_23.float()
        pow_34 = float_75.pow(2)
        mean_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_88 = mean_33 + 1e-06
        mean_33 = None
        rsqrt_33 = torch.rsqrt(add_88)
        add_88 = None
        output_66 = float_75 * rsqrt_33
        float_75 = rsqrt_33 = None
        float_76 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_89 = 1.0 + float_76
        float_76 = None
        output_67 = output_66 * add_89
        output_66 = add_89 = None
        hidden_states_43 = output_67.type_as(attn_output_23)
        output_67 = attn_output_23 = None
        hidden_states_44 = hidden_states_39 + hidden_states_43
        hidden_states_39 = hidden_states_43 = None
        float_77 = hidden_states_44.float()
        pow_35 = float_77.pow(2)
        mean_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        add_91 = mean_34 + 1e-06
        mean_34 = None
        rsqrt_34 = torch.rsqrt(add_91)
        add_91 = None
        output_68 = float_77 * rsqrt_34
        float_77 = rsqrt_34 = None
        float_78 = (
            l_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_92 = 1.0 + float_78
        float_78 = None
        output_69 = output_68 * add_92
        output_68 = add_92 = None
        hidden_states_45 = output_69.type_as(hidden_states_44)
        output_69 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_5 = torch._C._nn.gelu(linear_39, approximate="tanh")
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_45 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_104 = gelu_5 * linear_40
        gelu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_104,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_104 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_79 = down_proj_5.float()
        pow_36 = float_79.pow(2)
        mean_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_93 = mean_35 + 1e-06
        mean_35 = None
        rsqrt_35 = torch.rsqrt(add_93)
        add_93 = None
        output_70 = float_79 * rsqrt_35
        float_79 = rsqrt_35 = None
        float_80 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_94 = 1.0 + float_80
        float_80 = None
        output_71 = output_70 * add_94
        output_70 = add_94 = None
        hidden_states_46 = output_71.type_as(down_proj_5)
        output_71 = down_proj_5 = None
        hidden_states_47 = hidden_states_44 + hidden_states_46
        hidden_states_44 = hidden_states_46 = None
        float_81 = hidden_states_47.float()
        pow_37 = float_81.pow(2)
        mean_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_96 = mean_36 + 1e-06
        mean_36 = None
        rsqrt_36 = torch.rsqrt(add_96)
        add_96 = None
        output_72 = float_81 * rsqrt_36
        float_81 = rsqrt_36 = None
        float_82 = (
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_97 = 1.0 + float_82
        float_82 = None
        output_73 = output_72 * add_97
        output_72 = add_97 = None
        hidden_states_48 = output_73.type_as(hidden_states_47)
        output_73 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 20, -1, 256))
        linear_42 = None
        query_states_12 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 20, -1, 256))
        linear_43 = None
        key_states_12 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_48,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_48 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = linear_44.view((1, 20, -1, 256))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        float_83 = query_states_12.float()
        pow_38 = float_83.pow(2)
        mean_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_98 = mean_37 + 1e-06
        mean_37 = None
        rsqrt_37 = torch.rsqrt(add_98)
        add_98 = None
        output_74 = float_83 * rsqrt_37
        float_83 = rsqrt_37 = None
        float_84 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_99 = 1.0 + float_84
        float_84 = None
        output_75 = output_74 * add_99
        output_74 = add_99 = None
        query_states_13 = output_75.type_as(query_states_12)
        output_75 = query_states_12 = None
        float_85 = key_states_12.float()
        pow_39 = float_85.pow(2)
        mean_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        add_100 = mean_38 + 1e-06
        mean_38 = None
        rsqrt_38 = torch.rsqrt(add_100)
        add_100 = None
        output_76 = float_85 * rsqrt_38
        float_85 = rsqrt_38 = None
        float_86 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_101 = 1.0 + float_86
        float_86 = None
        output_77 = output_76 * add_101
        output_76 = add_101 = None
        key_states_13 = output_77.type_as(key_states_12)
        output_77 = key_states_12 = None
        cos_12 = cos_4.unsqueeze(1)
        sin_12 = sin_4.unsqueeze(1)
        mul_113 = query_states_13 * cos_12
        x1_12 = query_states_13[(Ellipsis, slice(None, 128, None))]
        x2_12 = query_states_13[(Ellipsis, slice(128, None, None))]
        query_states_13 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_14 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_114 = cat_14 * sin_12
        cat_14 = None
        q_embed_6 = mul_113 + mul_114
        mul_113 = mul_114 = None
        mul_115 = key_states_13 * cos_12
        cos_12 = None
        x1_13 = key_states_13[(Ellipsis, slice(None, 128, None))]
        x2_13 = key_states_13[(Ellipsis, slice(128, None, None))]
        key_states_13 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_15 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_116 = cat_15 * sin_12
        cat_15 = sin_12 = None
        k_embed_6 = mul_115 + mul_116
        mul_115 = mul_116 = None
        getitem_50 = k_embed_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_6 = None
        hidden_states_49 = getitem_50.expand(1, 1, 4, 20, 256)
        getitem_50 = None
        key_12 = hidden_states_49.reshape(1, 4, 20, 256)
        hidden_states_49 = None
        getitem_51 = value_states_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_6 = None
        hidden_states_50 = getitem_51.expand(1, 1, 4, 20, 256)
        getitem_51 = None
        value_12 = hidden_states_50.reshape(1, 4, 20, 256)
        hidden_states_50 = None
        attention_mask_8 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_6 = key_13 = value_13 = attention_mask_8 = None
        transpose_29 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_29.contiguous()
        transpose_29 = None
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
        float_87 = attn_output_27.float()
        pow_40 = float_87.pow(2)
        mean_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_104 = mean_39 + 1e-06
        mean_39 = None
        rsqrt_39 = torch.rsqrt(add_104)
        add_104 = None
        output_78 = float_87 * rsqrt_39
        float_87 = rsqrt_39 = None
        float_88 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_105 = 1.0 + float_88
        float_88 = None
        output_79 = output_78 * add_105
        output_78 = add_105 = None
        hidden_states_51 = output_79.type_as(attn_output_27)
        output_79 = attn_output_27 = None
        hidden_states_52 = hidden_states_47 + hidden_states_51
        hidden_states_47 = hidden_states_51 = None
        float_89 = hidden_states_52.float()
        pow_41 = float_89.pow(2)
        mean_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_107 = mean_40 + 1e-06
        mean_40 = None
        rsqrt_40 = torch.rsqrt(add_107)
        add_107 = None
        output_80 = float_89 * rsqrt_40
        float_89 = rsqrt_40 = None
        float_90 = (
            l_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_108 = 1.0 + float_90
        float_90 = None
        output_81 = output_80 * add_108
        output_80 = add_108 = None
        hidden_states_53 = output_81.type_as(hidden_states_52)
        output_81 = None
        linear_46 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_6 = torch._C._nn.gelu(linear_46, approximate="tanh")
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_53,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_53 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_121 = gelu_6 * linear_47
        gelu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_121,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_121 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_91 = down_proj_6.float()
        pow_42 = float_91.pow(2)
        mean_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_109 = mean_41 + 1e-06
        mean_41 = None
        rsqrt_41 = torch.rsqrt(add_109)
        add_109 = None
        output_82 = float_91 * rsqrt_41
        float_91 = rsqrt_41 = None
        float_92 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_110 = 1.0 + float_92
        float_92 = None
        output_83 = output_82 * add_110
        output_82 = add_110 = None
        hidden_states_54 = output_83.type_as(down_proj_6)
        output_83 = down_proj_6 = None
        hidden_states_55 = hidden_states_52 + hidden_states_54
        hidden_states_52 = hidden_states_54 = None
        float_93 = hidden_states_55.float()
        pow_43 = float_93.pow(2)
        mean_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        add_112 = mean_42 + 1e-06
        mean_42 = None
        rsqrt_42 = torch.rsqrt(add_112)
        add_112 = None
        output_84 = float_93 * rsqrt_42
        float_93 = rsqrt_42 = None
        float_94 = (
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_113 = 1.0 + float_94
        float_94 = None
        output_85 = output_84 * add_113
        output_84 = add_113 = None
        hidden_states_56 = output_85.type_as(hidden_states_55)
        output_85 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 20, -1, 256))
        linear_49 = None
        query_states_14 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 20, -1, 256))
        linear_50 = None
        key_states_14 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_56,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_56 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = linear_51.view((1, 20, -1, 256))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        float_95 = query_states_14.float()
        pow_44 = float_95.pow(2)
        mean_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_114 = mean_43 + 1e-06
        mean_43 = None
        rsqrt_43 = torch.rsqrt(add_114)
        add_114 = None
        output_86 = float_95 * rsqrt_43
        float_95 = rsqrt_43 = None
        float_96 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_115 = 1.0 + float_96
        float_96 = None
        output_87 = output_86 * add_115
        output_86 = add_115 = None
        query_states_15 = output_87.type_as(query_states_14)
        output_87 = query_states_14 = None
        float_97 = key_states_14.float()
        pow_45 = float_97.pow(2)
        mean_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_116 = mean_44 + 1e-06
        mean_44 = None
        rsqrt_44 = torch.rsqrt(add_116)
        add_116 = None
        output_88 = float_97 * rsqrt_44
        float_97 = rsqrt_44 = None
        float_98 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_117 = 1.0 + float_98
        float_98 = None
        output_89 = output_88 * add_117
        output_88 = add_117 = None
        key_states_15 = output_89.type_as(key_states_14)
        output_89 = key_states_14 = None
        cos_13 = cos_4.unsqueeze(1)
        sin_13 = sin_4.unsqueeze(1)
        mul_130 = query_states_15 * cos_13
        x1_14 = query_states_15[(Ellipsis, slice(None, 128, None))]
        x2_14 = query_states_15[(Ellipsis, slice(128, None, None))]
        query_states_15 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_16 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_131 = cat_16 * sin_13
        cat_16 = None
        q_embed_7 = mul_130 + mul_131
        mul_130 = mul_131 = None
        mul_132 = key_states_15 * cos_13
        cos_13 = None
        x1_15 = key_states_15[(Ellipsis, slice(None, 128, None))]
        x2_15 = key_states_15[(Ellipsis, slice(128, None, None))]
        key_states_15 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_17 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_133 = cat_17 * sin_13
        cat_17 = sin_13 = None
        k_embed_7 = mul_132 + mul_133
        mul_132 = mul_133 = None
        getitem_57 = k_embed_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_7 = None
        hidden_states_57 = getitem_57.expand(1, 1, 4, 20, 256)
        getitem_57 = None
        key_14 = hidden_states_57.reshape(1, 4, 20, 256)
        hidden_states_57 = None
        getitem_58 = value_states_7[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_7 = None
        hidden_states_58 = getitem_58.expand(1, 1, 4, 20, 256)
        getitem_58 = None
        value_14 = hidden_states_58.reshape(1, 4, 20, 256)
        hidden_states_58 = None
        attention_mask_9 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_7 = key_15 = value_15 = attention_mask_9 = None
        transpose_33 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_33.contiguous()
        transpose_33 = None
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
        float_99 = attn_output_31.float()
        pow_46 = float_99.pow(2)
        mean_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_120 = mean_45 + 1e-06
        mean_45 = None
        rsqrt_45 = torch.rsqrt(add_120)
        add_120 = None
        output_90 = float_99 * rsqrt_45
        float_99 = rsqrt_45 = None
        float_100 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_121 = 1.0 + float_100
        float_100 = None
        output_91 = output_90 * add_121
        output_90 = add_121 = None
        hidden_states_59 = output_91.type_as(attn_output_31)
        output_91 = attn_output_31 = None
        hidden_states_60 = hidden_states_55 + hidden_states_59
        hidden_states_55 = hidden_states_59 = None
        float_101 = hidden_states_60.float()
        pow_47 = float_101.pow(2)
        mean_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        add_123 = mean_46 + 1e-06
        mean_46 = None
        rsqrt_46 = torch.rsqrt(add_123)
        add_123 = None
        output_92 = float_101 * rsqrt_46
        float_101 = rsqrt_46 = None
        float_102 = (
            l_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_124 = 1.0 + float_102
        float_102 = None
        output_93 = output_92 * add_124
        output_92 = add_124 = None
        hidden_states_61 = output_93.type_as(hidden_states_60)
        output_93 = None
        linear_53 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_7 = torch._C._nn.gelu(linear_53, approximate="tanh")
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_61,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_61 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_138 = gelu_7 * linear_54
        gelu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_138,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_138 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_103 = down_proj_7.float()
        pow_48 = float_103.pow(2)
        mean_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_125 = mean_47 + 1e-06
        mean_47 = None
        rsqrt_47 = torch.rsqrt(add_125)
        add_125 = None
        output_94 = float_103 * rsqrt_47
        float_103 = rsqrt_47 = None
        float_104 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_126 = 1.0 + float_104
        float_104 = None
        output_95 = output_94 * add_126
        output_94 = add_126 = None
        hidden_states_62 = output_95.type_as(down_proj_7)
        output_95 = down_proj_7 = None
        hidden_states_63 = hidden_states_60 + hidden_states_62
        hidden_states_60 = hidden_states_62 = None
        float_105 = hidden_states_63.float()
        pow_49 = float_105.pow(2)
        mean_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_128 = mean_48 + 1e-06
        mean_48 = None
        rsqrt_48 = torch.rsqrt(add_128)
        add_128 = None
        output_96 = float_105 * rsqrt_48
        float_105 = rsqrt_48 = None
        float_106 = (
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_129 = 1.0 + float_106
        float_106 = None
        output_97 = output_96 * add_129
        output_96 = add_129 = None
        hidden_states_64 = output_97.type_as(hidden_states_63)
        output_97 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_56.view((1, 20, -1, 256))
        linear_56 = None
        query_states_16 = view_24.transpose(1, 2)
        view_24 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_57.view((1, 20, -1, 256))
        linear_57 = None
        key_states_16 = view_25.transpose(1, 2)
        view_25 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_64,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_64 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = linear_58.view((1, 20, -1, 256))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        float_107 = query_states_16.float()
        pow_50 = float_107.pow(2)
        mean_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_130 = mean_49 + 1e-06
        mean_49 = None
        rsqrt_49 = torch.rsqrt(add_130)
        add_130 = None
        output_98 = float_107 * rsqrt_49
        float_107 = rsqrt_49 = None
        float_108 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_131 = 1.0 + float_108
        float_108 = None
        output_99 = output_98 * add_131
        output_98 = add_131 = None
        query_states_17 = output_99.type_as(query_states_16)
        output_99 = query_states_16 = None
        float_109 = key_states_16.float()
        pow_51 = float_109.pow(2)
        mean_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        add_132 = mean_50 + 1e-06
        mean_50 = None
        rsqrt_50 = torch.rsqrt(add_132)
        add_132 = None
        output_100 = float_109 * rsqrt_50
        float_109 = rsqrt_50 = None
        float_110 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_133 = 1.0 + float_110
        float_110 = None
        output_101 = output_100 * add_133
        output_100 = add_133 = None
        key_states_17 = output_101.type_as(key_states_16)
        output_101 = key_states_16 = None
        cos_14 = cos_4.unsqueeze(1)
        sin_14 = sin_4.unsqueeze(1)
        mul_147 = query_states_17 * cos_14
        x1_16 = query_states_17[(Ellipsis, slice(None, 128, None))]
        x2_16 = query_states_17[(Ellipsis, slice(128, None, None))]
        query_states_17 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_18 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_148 = cat_18 * sin_14
        cat_18 = None
        q_embed_8 = mul_147 + mul_148
        mul_147 = mul_148 = None
        mul_149 = key_states_17 * cos_14
        cos_14 = None
        x1_17 = key_states_17[(Ellipsis, slice(None, 128, None))]
        x2_17 = key_states_17[(Ellipsis, slice(128, None, None))]
        key_states_17 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_19 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_150 = cat_19 * sin_14
        cat_19 = sin_14 = None
        k_embed_8 = mul_149 + mul_150
        mul_149 = mul_150 = None
        getitem_64 = k_embed_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_8 = None
        hidden_states_65 = getitem_64.expand(1, 1, 4, 20, 256)
        getitem_64 = None
        key_16 = hidden_states_65.reshape(1, 4, 20, 256)
        hidden_states_65 = None
        getitem_65 = value_states_8[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_8 = None
        hidden_states_66 = getitem_65.expand(1, 1, 4, 20, 256)
        getitem_65 = None
        value_16 = hidden_states_66.reshape(1, 4, 20, 256)
        hidden_states_66 = None
        attention_mask_10 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_8 = key_17 = value_17 = attention_mask_10 = None
        transpose_37 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_37.contiguous()
        transpose_37 = None
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
        float_111 = attn_output_35.float()
        pow_52 = float_111.pow(2)
        mean_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_136 = mean_51 + 1e-06
        mean_51 = None
        rsqrt_51 = torch.rsqrt(add_136)
        add_136 = None
        output_102 = float_111 * rsqrt_51
        float_111 = rsqrt_51 = None
        float_112 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_137 = 1.0 + float_112
        float_112 = None
        output_103 = output_102 * add_137
        output_102 = add_137 = None
        hidden_states_67 = output_103.type_as(attn_output_35)
        output_103 = attn_output_35 = None
        hidden_states_68 = hidden_states_63 + hidden_states_67
        hidden_states_63 = hidden_states_67 = None
        float_113 = hidden_states_68.float()
        pow_53 = float_113.pow(2)
        mean_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_139 = mean_52 + 1e-06
        mean_52 = None
        rsqrt_52 = torch.rsqrt(add_139)
        add_139 = None
        output_104 = float_113 * rsqrt_52
        float_113 = rsqrt_52 = None
        float_114 = (
            l_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_140 = 1.0 + float_114
        float_114 = None
        output_105 = output_104 * add_140
        output_104 = add_140 = None
        hidden_states_69 = output_105.type_as(hidden_states_68)
        output_105 = None
        linear_60 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_8 = torch._C._nn.gelu(linear_60, approximate="tanh")
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_69 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_155 = gelu_8 * linear_61
        gelu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_155,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_155 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_115 = down_proj_8.float()
        pow_54 = float_115.pow(2)
        mean_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_141 = mean_53 + 1e-06
        mean_53 = None
        rsqrt_53 = torch.rsqrt(add_141)
        add_141 = None
        output_106 = float_115 * rsqrt_53
        float_115 = rsqrt_53 = None
        float_116 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_142 = 1.0 + float_116
        float_116 = None
        output_107 = output_106 * add_142
        output_106 = add_142 = None
        hidden_states_70 = output_107.type_as(down_proj_8)
        output_107 = down_proj_8 = None
        hidden_states_71 = hidden_states_68 + hidden_states_70
        hidden_states_68 = hidden_states_70 = None
        float_117 = hidden_states_71.float()
        pow_55 = float_117.pow(2)
        mean_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        add_144 = mean_54 + 1e-06
        mean_54 = None
        rsqrt_54 = torch.rsqrt(add_144)
        add_144 = None
        output_108 = float_117 * rsqrt_54
        float_117 = rsqrt_54 = None
        float_118 = (
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_145 = 1.0 + float_118
        float_118 = None
        output_109 = output_108 * add_145
        output_108 = add_145 = None
        hidden_states_72 = output_109.type_as(hidden_states_71)
        output_109 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_63.view((1, 20, -1, 256))
        linear_63 = None
        query_states_18 = view_27.transpose(1, 2)
        view_27 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_64.view((1, 20, -1, 256))
        linear_64 = None
        key_states_18 = view_28.transpose(1, 2)
        view_28 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = linear_65.view((1, 20, -1, 256))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        float_119 = query_states_18.float()
        pow_56 = float_119.pow(2)
        mean_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_146 = mean_55 + 1e-06
        mean_55 = None
        rsqrt_55 = torch.rsqrt(add_146)
        add_146 = None
        output_110 = float_119 * rsqrt_55
        float_119 = rsqrt_55 = None
        float_120 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_147 = 1.0 + float_120
        float_120 = None
        output_111 = output_110 * add_147
        output_110 = add_147 = None
        query_states_19 = output_111.type_as(query_states_18)
        output_111 = query_states_18 = None
        float_121 = key_states_18.float()
        pow_57 = float_121.pow(2)
        mean_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_148 = mean_56 + 1e-06
        mean_56 = None
        rsqrt_56 = torch.rsqrt(add_148)
        add_148 = None
        output_112 = float_121 * rsqrt_56
        float_121 = rsqrt_56 = None
        float_122 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_149 = 1.0 + float_122
        float_122 = None
        output_113 = output_112 * add_149
        output_112 = add_149 = None
        key_states_19 = output_113.type_as(key_states_18)
        output_113 = key_states_18 = None
        cos_15 = cos_4.unsqueeze(1)
        sin_15 = sin_4.unsqueeze(1)
        mul_164 = query_states_19 * cos_15
        x1_18 = query_states_19[(Ellipsis, slice(None, 128, None))]
        x2_18 = query_states_19[(Ellipsis, slice(128, None, None))]
        query_states_19 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_20 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_165 = cat_20 * sin_15
        cat_20 = None
        q_embed_9 = mul_164 + mul_165
        mul_164 = mul_165 = None
        mul_166 = key_states_19 * cos_15
        cos_15 = None
        x1_19 = key_states_19[(Ellipsis, slice(None, 128, None))]
        x2_19 = key_states_19[(Ellipsis, slice(128, None, None))]
        key_states_19 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_21 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_167 = cat_21 * sin_15
        cat_21 = sin_15 = None
        k_embed_9 = mul_166 + mul_167
        mul_166 = mul_167 = None
        getitem_71 = k_embed_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_9 = None
        hidden_states_73 = getitem_71.expand(1, 1, 4, 20, 256)
        getitem_71 = None
        key_18 = hidden_states_73.reshape(1, 4, 20, 256)
        hidden_states_73 = None
        getitem_72 = value_states_9[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_9 = None
        hidden_states_74 = getitem_72.expand(1, 1, 4, 20, 256)
        getitem_72 = None
        value_18 = hidden_states_74.reshape(1, 4, 20, 256)
        hidden_states_74 = None
        attention_mask_11 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_9 = key_19 = value_19 = attention_mask_11 = None
        transpose_41 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_41.contiguous()
        transpose_41 = None
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
        float_123 = attn_output_39.float()
        pow_58 = float_123.pow(2)
        mean_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_152 = mean_57 + 1e-06
        mean_57 = None
        rsqrt_57 = torch.rsqrt(add_152)
        add_152 = None
        output_114 = float_123 * rsqrt_57
        float_123 = rsqrt_57 = None
        float_124 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_153 = 1.0 + float_124
        float_124 = None
        output_115 = output_114 * add_153
        output_114 = add_153 = None
        hidden_states_75 = output_115.type_as(attn_output_39)
        output_115 = attn_output_39 = None
        hidden_states_76 = hidden_states_71 + hidden_states_75
        hidden_states_71 = hidden_states_75 = None
        float_125 = hidden_states_76.float()
        pow_59 = float_125.pow(2)
        mean_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        add_155 = mean_58 + 1e-06
        mean_58 = None
        rsqrt_58 = torch.rsqrt(add_155)
        add_155 = None
        output_116 = float_125 * rsqrt_58
        float_125 = rsqrt_58 = None
        float_126 = (
            l_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_156 = 1.0 + float_126
        float_126 = None
        output_117 = output_116 * add_156
        output_116 = add_156 = None
        hidden_states_77 = output_117.type_as(hidden_states_76)
        output_117 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_9 = torch._C._nn.gelu(linear_67, approximate="tanh")
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_77 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_172 = gelu_9 * linear_68
        gelu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_172,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_172 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_127 = down_proj_9.float()
        pow_60 = float_127.pow(2)
        mean_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_157 = mean_59 + 1e-06
        mean_59 = None
        rsqrt_59 = torch.rsqrt(add_157)
        add_157 = None
        output_118 = float_127 * rsqrt_59
        float_127 = rsqrt_59 = None
        float_128 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_158 = 1.0 + float_128
        float_128 = None
        output_119 = output_118 * add_158
        output_118 = add_158 = None
        hidden_states_78 = output_119.type_as(down_proj_9)
        output_119 = down_proj_9 = None
        hidden_states_79 = hidden_states_76 + hidden_states_78
        hidden_states_76 = hidden_states_78 = None
        float_129 = hidden_states_79.float()
        pow_61 = float_129.pow(2)
        mean_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_160 = mean_60 + 1e-06
        mean_60 = None
        rsqrt_60 = torch.rsqrt(add_160)
        add_160 = None
        output_120 = float_129 * rsqrt_60
        float_129 = rsqrt_60 = None
        float_130 = (
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_161 = 1.0 + float_130
        float_130 = None
        output_121 = output_120 * add_161
        output_120 = add_161 = None
        hidden_states_80 = output_121.type_as(hidden_states_79)
        output_121 = None
        linear_70 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_70.view((1, 20, -1, 256))
        linear_70 = None
        query_states_20 = view_30.transpose(1, 2)
        view_30 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_71.view((1, 20, -1, 256))
        linear_71 = None
        key_states_20 = view_31.transpose(1, 2)
        view_31 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_80 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = linear_72.view((1, 20, -1, 256))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        float_131 = query_states_20.float()
        pow_62 = float_131.pow(2)
        mean_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_162 = mean_61 + 1e-06
        mean_61 = None
        rsqrt_61 = torch.rsqrt(add_162)
        add_162 = None
        output_122 = float_131 * rsqrt_61
        float_131 = rsqrt_61 = None
        float_132 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_163 = 1.0 + float_132
        float_132 = None
        output_123 = output_122 * add_163
        output_122 = add_163 = None
        query_states_21 = output_123.type_as(query_states_20)
        output_123 = query_states_20 = None
        float_133 = key_states_20.float()
        pow_63 = float_133.pow(2)
        mean_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        add_164 = mean_62 + 1e-06
        mean_62 = None
        rsqrt_62 = torch.rsqrt(add_164)
        add_164 = None
        output_124 = float_133 * rsqrt_62
        float_133 = rsqrt_62 = None
        float_134 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_165 = 1.0 + float_134
        float_134 = None
        output_125 = output_124 * add_165
        output_124 = add_165 = None
        key_states_21 = output_125.type_as(key_states_20)
        output_125 = key_states_20 = None
        cos_16 = cos_4.unsqueeze(1)
        sin_16 = sin_4.unsqueeze(1)
        mul_181 = query_states_21 * cos_16
        x1_20 = query_states_21[(Ellipsis, slice(None, 128, None))]
        x2_20 = query_states_21[(Ellipsis, slice(128, None, None))]
        query_states_21 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_22 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_182 = cat_22 * sin_16
        cat_22 = None
        q_embed_10 = mul_181 + mul_182
        mul_181 = mul_182 = None
        mul_183 = key_states_21 * cos_16
        cos_16 = None
        x1_21 = key_states_21[(Ellipsis, slice(None, 128, None))]
        x2_21 = key_states_21[(Ellipsis, slice(128, None, None))]
        key_states_21 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_23 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_184 = cat_23 * sin_16
        cat_23 = sin_16 = None
        k_embed_10 = mul_183 + mul_184
        mul_183 = mul_184 = None
        getitem_78 = k_embed_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_10 = None
        hidden_states_81 = getitem_78.expand(1, 1, 4, 20, 256)
        getitem_78 = None
        key_20 = hidden_states_81.reshape(1, 4, 20, 256)
        hidden_states_81 = None
        getitem_79 = value_states_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_10 = None
        hidden_states_82 = getitem_79.expand(1, 1, 4, 20, 256)
        getitem_79 = None
        value_20 = hidden_states_82.reshape(1, 4, 20, 256)
        hidden_states_82 = None
        attention_mask_12 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_10 = key_21 = value_21 = attention_mask_12 = None
        transpose_45 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_45.contiguous()
        transpose_45 = None
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
        float_135 = attn_output_43.float()
        pow_64 = float_135.pow(2)
        mean_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_168 = mean_63 + 1e-06
        mean_63 = None
        rsqrt_63 = torch.rsqrt(add_168)
        add_168 = None
        output_126 = float_135 * rsqrt_63
        float_135 = rsqrt_63 = None
        float_136 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_169 = 1.0 + float_136
        float_136 = None
        output_127 = output_126 * add_169
        output_126 = add_169 = None
        hidden_states_83 = output_127.type_as(attn_output_43)
        output_127 = attn_output_43 = None
        hidden_states_84 = hidden_states_79 + hidden_states_83
        hidden_states_79 = hidden_states_83 = None
        float_137 = hidden_states_84.float()
        pow_65 = float_137.pow(2)
        mean_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_171 = mean_64 + 1e-06
        mean_64 = None
        rsqrt_64 = torch.rsqrt(add_171)
        add_171 = None
        output_128 = float_137 * rsqrt_64
        float_137 = rsqrt_64 = None
        float_138 = (
            l_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_172 = 1.0 + float_138
        float_138 = None
        output_129 = output_128 * add_172
        output_128 = add_172 = None
        hidden_states_85 = output_129.type_as(hidden_states_84)
        output_129 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_10 = torch._C._nn.gelu(linear_74, approximate="tanh")
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_85 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_189 = gelu_10 * linear_75
        gelu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_189,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_189 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_139 = down_proj_10.float()
        pow_66 = float_139.pow(2)
        mean_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_173 = mean_65 + 1e-06
        mean_65 = None
        rsqrt_65 = torch.rsqrt(add_173)
        add_173 = None
        output_130 = float_139 * rsqrt_65
        float_139 = rsqrt_65 = None
        float_140 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_174 = 1.0 + float_140
        float_140 = None
        output_131 = output_130 * add_174
        output_130 = add_174 = None
        hidden_states_86 = output_131.type_as(down_proj_10)
        output_131 = down_proj_10 = None
        hidden_states_87 = hidden_states_84 + hidden_states_86
        hidden_states_84 = hidden_states_86 = None
        float_141 = hidden_states_87.float()
        pow_67 = float_141.pow(2)
        mean_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        add_176 = mean_66 + 1e-06
        mean_66 = None
        rsqrt_66 = torch.rsqrt(add_176)
        add_176 = None
        output_132 = float_141 * rsqrt_66
        float_141 = rsqrt_66 = None
        float_142 = (
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_177 = 1.0 + float_142
        float_142 = None
        output_133 = output_132 * add_177
        output_132 = add_177 = None
        hidden_states_88 = output_133.type_as(hidden_states_87)
        output_133 = None
        linear_77 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_77.view((1, 20, -1, 256))
        linear_77 = None
        query_states_22 = view_33.transpose(1, 2)
        view_33 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_78.view((1, 20, -1, 256))
        linear_78 = None
        key_states_22 = view_34.transpose(1, 2)
        view_34 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_88,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_88 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = linear_79.view((1, 20, -1, 256))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        float_143 = query_states_22.float()
        pow_68 = float_143.pow(2)
        mean_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_178 = mean_67 + 1e-06
        mean_67 = None
        rsqrt_67 = torch.rsqrt(add_178)
        add_178 = None
        output_134 = float_143 * rsqrt_67
        float_143 = rsqrt_67 = None
        float_144 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_179 = 1.0 + float_144
        float_144 = None
        output_135 = output_134 * add_179
        output_134 = add_179 = None
        query_states_23 = output_135.type_as(query_states_22)
        output_135 = query_states_22 = None
        float_145 = key_states_22.float()
        pow_69 = float_145.pow(2)
        mean_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_180 = mean_68 + 1e-06
        mean_68 = None
        rsqrt_68 = torch.rsqrt(add_180)
        add_180 = None
        output_136 = float_145 * rsqrt_68
        float_145 = rsqrt_68 = None
        float_146 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_181 = 1.0 + float_146
        float_146 = None
        output_137 = output_136 * add_181
        output_136 = add_181 = None
        key_states_23 = output_137.type_as(key_states_22)
        output_137 = key_states_22 = None
        cos_17 = cos_10.unsqueeze(1)
        sin_17 = sin_10.unsqueeze(1)
        mul_198 = query_states_23 * cos_17
        x1_22 = query_states_23[(Ellipsis, slice(None, 128, None))]
        x2_22 = query_states_23[(Ellipsis, slice(128, None, None))]
        query_states_23 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_24 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_199 = cat_24 * sin_17
        cat_24 = None
        q_embed_11 = mul_198 + mul_199
        mul_198 = mul_199 = None
        mul_200 = key_states_23 * cos_17
        cos_17 = None
        x1_23 = key_states_23[(Ellipsis, slice(None, 128, None))]
        x2_23 = key_states_23[(Ellipsis, slice(128, None, None))]
        key_states_23 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_25 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_201 = cat_25 * sin_17
        cat_25 = sin_17 = None
        k_embed_11 = mul_200 + mul_201
        mul_200 = mul_201 = None
        getitem_85 = k_embed_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_11 = None
        hidden_states_89 = getitem_85.expand(1, 1, 4, 20, 256)
        getitem_85 = None
        key_22 = hidden_states_89.reshape(1, 4, 20, 256)
        hidden_states_89 = None
        getitem_86 = value_states_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_11 = None
        hidden_states_90 = getitem_86.expand(1, 1, 4, 20, 256)
        getitem_86 = None
        value_22 = hidden_states_90.reshape(1, 4, 20, 256)
        hidden_states_90 = None
        attention_mask_13 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_11 = key_23 = value_23 = attention_mask_13 = None
        transpose_49 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_49.contiguous()
        transpose_49 = None
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
        float_147 = attn_output_47.float()
        pow_70 = float_147.pow(2)
        mean_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_184 = mean_69 + 1e-06
        mean_69 = None
        rsqrt_69 = torch.rsqrt(add_184)
        add_184 = None
        output_138 = float_147 * rsqrt_69
        float_147 = rsqrt_69 = None
        float_148 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_185 = 1.0 + float_148
        float_148 = None
        output_139 = output_138 * add_185
        output_138 = add_185 = None
        hidden_states_91 = output_139.type_as(attn_output_47)
        output_139 = attn_output_47 = None
        hidden_states_92 = hidden_states_87 + hidden_states_91
        hidden_states_87 = hidden_states_91 = None
        float_149 = hidden_states_92.float()
        pow_71 = float_149.pow(2)
        mean_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        add_187 = mean_70 + 1e-06
        mean_70 = None
        rsqrt_70 = torch.rsqrt(add_187)
        add_187 = None
        output_140 = float_149 * rsqrt_70
        float_149 = rsqrt_70 = None
        float_150 = (
            l_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_188 = 1.0 + float_150
        float_150 = None
        output_141 = output_140 * add_188
        output_140 = add_188 = None
        hidden_states_93 = output_141.type_as(hidden_states_92)
        output_141 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_11 = torch._C._nn.gelu(linear_81, approximate="tanh")
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_93 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_206 = gelu_11 * linear_82
        gelu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_206,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_206 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_151 = down_proj_11.float()
        pow_72 = float_151.pow(2)
        mean_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_189 = mean_71 + 1e-06
        mean_71 = None
        rsqrt_71 = torch.rsqrt(add_189)
        add_189 = None
        output_142 = float_151 * rsqrt_71
        float_151 = rsqrt_71 = None
        float_152 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_190 = 1.0 + float_152
        float_152 = None
        output_143 = output_142 * add_190
        output_142 = add_190 = None
        hidden_states_94 = output_143.type_as(down_proj_11)
        output_143 = down_proj_11 = None
        hidden_states_95 = hidden_states_92 + hidden_states_94
        hidden_states_92 = hidden_states_94 = None
        float_153 = hidden_states_95.float()
        pow_73 = float_153.pow(2)
        mean_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_192 = mean_72 + 1e-06
        mean_72 = None
        rsqrt_72 = torch.rsqrt(add_192)
        add_192 = None
        output_144 = float_153 * rsqrt_72
        float_153 = rsqrt_72 = None
        float_154 = (
            l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_193 = 1.0 + float_154
        float_154 = None
        output_145 = output_144 * add_193
        output_144 = add_193 = None
        hidden_states_96 = output_145.type_as(hidden_states_95)
        output_145 = None
        linear_84 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_36 = linear_84.view((1, 20, -1, 256))
        linear_84 = None
        query_states_24 = view_36.transpose(1, 2)
        view_36 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_85.view((1, 20, -1, 256))
        linear_85 = None
        key_states_24 = view_37.transpose(1, 2)
        view_37 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_96,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_96 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_38 = linear_86.view((1, 20, -1, 256))
        linear_86 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        float_155 = query_states_24.float()
        pow_74 = float_155.pow(2)
        mean_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_194 = mean_73 + 1e-06
        mean_73 = None
        rsqrt_73 = torch.rsqrt(add_194)
        add_194 = None
        output_146 = float_155 * rsqrt_73
        float_155 = rsqrt_73 = None
        float_156 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_195 = 1.0 + float_156
        float_156 = None
        output_147 = output_146 * add_195
        output_146 = add_195 = None
        query_states_25 = output_147.type_as(query_states_24)
        output_147 = query_states_24 = None
        float_157 = key_states_24.float()
        pow_75 = float_157.pow(2)
        mean_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        add_196 = mean_74 + 1e-06
        mean_74 = None
        rsqrt_74 = torch.rsqrt(add_196)
        add_196 = None
        output_148 = float_157 * rsqrt_74
        float_157 = rsqrt_74 = None
        float_158 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_197 = 1.0 + float_158
        float_158 = None
        output_149 = output_148 * add_197
        output_148 = add_197 = None
        key_states_25 = output_149.type_as(key_states_24)
        output_149 = key_states_24 = None
        cos_18 = cos_4.unsqueeze(1)
        sin_18 = sin_4.unsqueeze(1)
        mul_215 = query_states_25 * cos_18
        x1_24 = query_states_25[(Ellipsis, slice(None, 128, None))]
        x2_24 = query_states_25[(Ellipsis, slice(128, None, None))]
        query_states_25 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_26 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_216 = cat_26 * sin_18
        cat_26 = None
        q_embed_12 = mul_215 + mul_216
        mul_215 = mul_216 = None
        mul_217 = key_states_25 * cos_18
        cos_18 = None
        x1_25 = key_states_25[(Ellipsis, slice(None, 128, None))]
        x2_25 = key_states_25[(Ellipsis, slice(128, None, None))]
        key_states_25 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_27 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_218 = cat_27 * sin_18
        cat_27 = sin_18 = None
        k_embed_12 = mul_217 + mul_218
        mul_217 = mul_218 = None
        getitem_92 = k_embed_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_12 = None
        hidden_states_97 = getitem_92.expand(1, 1, 4, 20, 256)
        getitem_92 = None
        key_24 = hidden_states_97.reshape(1, 4, 20, 256)
        hidden_states_97 = None
        getitem_93 = value_states_12[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_12 = None
        hidden_states_98 = getitem_93.expand(1, 1, 4, 20, 256)
        getitem_93 = None
        value_24 = hidden_states_98.reshape(1, 4, 20, 256)
        hidden_states_98 = None
        attention_mask_14 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_12 = key_25 = value_25 = attention_mask_14 = None
        transpose_53 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_53.contiguous()
        transpose_53 = None
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
        float_159 = attn_output_51.float()
        pow_76 = float_159.pow(2)
        mean_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_200 = mean_75 + 1e-06
        mean_75 = None
        rsqrt_75 = torch.rsqrt(add_200)
        add_200 = None
        output_150 = float_159 * rsqrt_75
        float_159 = rsqrt_75 = None
        float_160 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_201 = 1.0 + float_160
        float_160 = None
        output_151 = output_150 * add_201
        output_150 = add_201 = None
        hidden_states_99 = output_151.type_as(attn_output_51)
        output_151 = attn_output_51 = None
        hidden_states_100 = hidden_states_95 + hidden_states_99
        hidden_states_95 = hidden_states_99 = None
        float_161 = hidden_states_100.float()
        pow_77 = float_161.pow(2)
        mean_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_203 = mean_76 + 1e-06
        mean_76 = None
        rsqrt_76 = torch.rsqrt(add_203)
        add_203 = None
        output_152 = float_161 * rsqrt_76
        float_161 = rsqrt_76 = None
        float_162 = (
            l_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_204 = 1.0 + float_162
        float_162 = None
        output_153 = output_152 * add_204
        output_152 = add_204 = None
        hidden_states_101 = output_153.type_as(hidden_states_100)
        output_153 = None
        linear_88 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_12 = torch._C._nn.gelu(linear_88, approximate="tanh")
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_101,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_101 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_223 = gelu_12 * linear_89
        gelu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_223,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_223 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_163 = down_proj_12.float()
        pow_78 = float_163.pow(2)
        mean_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_205 = mean_77 + 1e-06
        mean_77 = None
        rsqrt_77 = torch.rsqrt(add_205)
        add_205 = None
        output_154 = float_163 * rsqrt_77
        float_163 = rsqrt_77 = None
        float_164 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_206 = 1.0 + float_164
        float_164 = None
        output_155 = output_154 * add_206
        output_154 = add_206 = None
        hidden_states_102 = output_155.type_as(down_proj_12)
        output_155 = down_proj_12 = None
        hidden_states_103 = hidden_states_100 + hidden_states_102
        hidden_states_100 = hidden_states_102 = None
        float_165 = hidden_states_103.float()
        pow_79 = float_165.pow(2)
        mean_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        add_208 = mean_78 + 1e-06
        mean_78 = None
        rsqrt_78 = torch.rsqrt(add_208)
        add_208 = None
        output_156 = float_165 * rsqrt_78
        float_165 = rsqrt_78 = None
        float_166 = (
            l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_209 = 1.0 + float_166
        float_166 = None
        output_157 = output_156 * add_209
        output_156 = add_209 = None
        hidden_states_104 = output_157.type_as(hidden_states_103)
        output_157 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_39 = linear_91.view((1, 20, -1, 256))
        linear_91 = None
        query_states_26 = view_39.transpose(1, 2)
        view_39 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_92.view((1, 20, -1, 256))
        linear_92 = None
        key_states_26 = view_40.transpose(1, 2)
        view_40 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_104,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_104 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_41 = linear_93.view((1, 20, -1, 256))
        linear_93 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        float_167 = query_states_26.float()
        pow_80 = float_167.pow(2)
        mean_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_210 = mean_79 + 1e-06
        mean_79 = None
        rsqrt_79 = torch.rsqrt(add_210)
        add_210 = None
        output_158 = float_167 * rsqrt_79
        float_167 = rsqrt_79 = None
        float_168 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_211 = 1.0 + float_168
        float_168 = None
        output_159 = output_158 * add_211
        output_158 = add_211 = None
        query_states_27 = output_159.type_as(query_states_26)
        output_159 = query_states_26 = None
        float_169 = key_states_26.float()
        pow_81 = float_169.pow(2)
        mean_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_212 = mean_80 + 1e-06
        mean_80 = None
        rsqrt_80 = torch.rsqrt(add_212)
        add_212 = None
        output_160 = float_169 * rsqrt_80
        float_169 = rsqrt_80 = None
        float_170 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_213 = 1.0 + float_170
        float_170 = None
        output_161 = output_160 * add_213
        output_160 = add_213 = None
        key_states_27 = output_161.type_as(key_states_26)
        output_161 = key_states_26 = None
        cos_19 = cos_4.unsqueeze(1)
        sin_19 = sin_4.unsqueeze(1)
        mul_232 = query_states_27 * cos_19
        x1_26 = query_states_27[(Ellipsis, slice(None, 128, None))]
        x2_26 = query_states_27[(Ellipsis, slice(128, None, None))]
        query_states_27 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_28 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_233 = cat_28 * sin_19
        cat_28 = None
        q_embed_13 = mul_232 + mul_233
        mul_232 = mul_233 = None
        mul_234 = key_states_27 * cos_19
        cos_19 = None
        x1_27 = key_states_27[(Ellipsis, slice(None, 128, None))]
        x2_27 = key_states_27[(Ellipsis, slice(128, None, None))]
        key_states_27 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_29 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_235 = cat_29 * sin_19
        cat_29 = sin_19 = None
        k_embed_13 = mul_234 + mul_235
        mul_234 = mul_235 = None
        getitem_99 = k_embed_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_13 = None
        hidden_states_105 = getitem_99.expand(1, 1, 4, 20, 256)
        getitem_99 = None
        key_26 = hidden_states_105.reshape(1, 4, 20, 256)
        hidden_states_105 = None
        getitem_100 = value_states_13[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_13 = None
        hidden_states_106 = getitem_100.expand(1, 1, 4, 20, 256)
        getitem_100 = None
        value_26 = hidden_states_106.reshape(1, 4, 20, 256)
        hidden_states_106 = None
        attention_mask_15 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_13 = key_27 = value_27 = attention_mask_15 = None
        transpose_57 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_57.contiguous()
        transpose_57 = None
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
        float_171 = attn_output_55.float()
        pow_82 = float_171.pow(2)
        mean_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_216 = mean_81 + 1e-06
        mean_81 = None
        rsqrt_81 = torch.rsqrt(add_216)
        add_216 = None
        output_162 = float_171 * rsqrt_81
        float_171 = rsqrt_81 = None
        float_172 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_217 = 1.0 + float_172
        float_172 = None
        output_163 = output_162 * add_217
        output_162 = add_217 = None
        hidden_states_107 = output_163.type_as(attn_output_55)
        output_163 = attn_output_55 = None
        hidden_states_108 = hidden_states_103 + hidden_states_107
        hidden_states_103 = hidden_states_107 = None
        float_173 = hidden_states_108.float()
        pow_83 = float_173.pow(2)
        mean_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        add_219 = mean_82 + 1e-06
        mean_82 = None
        rsqrt_82 = torch.rsqrt(add_219)
        add_219 = None
        output_164 = float_173 * rsqrt_82
        float_173 = rsqrt_82 = None
        float_174 = (
            l_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_220 = 1.0 + float_174
        float_174 = None
        output_165 = output_164 * add_220
        output_164 = add_220 = None
        hidden_states_109 = output_165.type_as(hidden_states_108)
        output_165 = None
        linear_95 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_13 = torch._C._nn.gelu(linear_95, approximate="tanh")
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_109,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_109 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_240 = gelu_13 * linear_96
        gelu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_240,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_240 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_175 = down_proj_13.float()
        pow_84 = float_175.pow(2)
        mean_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_221 = mean_83 + 1e-06
        mean_83 = None
        rsqrt_83 = torch.rsqrt(add_221)
        add_221 = None
        output_166 = float_175 * rsqrt_83
        float_175 = rsqrt_83 = None
        float_176 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_222 = 1.0 + float_176
        float_176 = None
        output_167 = output_166 * add_222
        output_166 = add_222 = None
        hidden_states_110 = output_167.type_as(down_proj_13)
        output_167 = down_proj_13 = None
        hidden_states_111 = hidden_states_108 + hidden_states_110
        hidden_states_108 = hidden_states_110 = None
        float_177 = hidden_states_111.float()
        pow_85 = float_177.pow(2)
        mean_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_224 = mean_84 + 1e-06
        mean_84 = None
        rsqrt_84 = torch.rsqrt(add_224)
        add_224 = None
        output_168 = float_177 * rsqrt_84
        float_177 = rsqrt_84 = None
        float_178 = (
            l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_225 = 1.0 + float_178
        float_178 = None
        output_169 = output_168 * add_225
        output_168 = add_225 = None
        hidden_states_112 = output_169.type_as(hidden_states_111)
        output_169 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_42 = linear_98.view((1, 20, -1, 256))
        linear_98 = None
        query_states_28 = view_42.transpose(1, 2)
        view_42 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_99.view((1, 20, -1, 256))
        linear_99 = None
        key_states_28 = view_43.transpose(1, 2)
        view_43 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_112 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_44 = linear_100.view((1, 20, -1, 256))
        linear_100 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        float_179 = query_states_28.float()
        pow_86 = float_179.pow(2)
        mean_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_226 = mean_85 + 1e-06
        mean_85 = None
        rsqrt_85 = torch.rsqrt(add_226)
        add_226 = None
        output_170 = float_179 * rsqrt_85
        float_179 = rsqrt_85 = None
        float_180 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_227 = 1.0 + float_180
        float_180 = None
        output_171 = output_170 * add_227
        output_170 = add_227 = None
        query_states_29 = output_171.type_as(query_states_28)
        output_171 = query_states_28 = None
        float_181 = key_states_28.float()
        pow_87 = float_181.pow(2)
        mean_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        add_228 = mean_86 + 1e-06
        mean_86 = None
        rsqrt_86 = torch.rsqrt(add_228)
        add_228 = None
        output_172 = float_181 * rsqrt_86
        float_181 = rsqrt_86 = None
        float_182 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_229 = 1.0 + float_182
        float_182 = None
        output_173 = output_172 * add_229
        output_172 = add_229 = None
        key_states_29 = output_173.type_as(key_states_28)
        output_173 = key_states_28 = None
        cos_20 = cos_4.unsqueeze(1)
        sin_20 = sin_4.unsqueeze(1)
        mul_249 = query_states_29 * cos_20
        x1_28 = query_states_29[(Ellipsis, slice(None, 128, None))]
        x2_28 = query_states_29[(Ellipsis, slice(128, None, None))]
        query_states_29 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_30 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_250 = cat_30 * sin_20
        cat_30 = None
        q_embed_14 = mul_249 + mul_250
        mul_249 = mul_250 = None
        mul_251 = key_states_29 * cos_20
        cos_20 = None
        x1_29 = key_states_29[(Ellipsis, slice(None, 128, None))]
        x2_29 = key_states_29[(Ellipsis, slice(128, None, None))]
        key_states_29 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_31 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_252 = cat_31 * sin_20
        cat_31 = sin_20 = None
        k_embed_14 = mul_251 + mul_252
        mul_251 = mul_252 = None
        getitem_106 = k_embed_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_14 = None
        hidden_states_113 = getitem_106.expand(1, 1, 4, 20, 256)
        getitem_106 = None
        key_28 = hidden_states_113.reshape(1, 4, 20, 256)
        hidden_states_113 = None
        getitem_107 = value_states_14[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_14 = None
        hidden_states_114 = getitem_107.expand(1, 1, 4, 20, 256)
        getitem_107 = None
        value_28 = hidden_states_114.reshape(1, 4, 20, 256)
        hidden_states_114 = None
        attention_mask_16 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_14 = key_29 = value_29 = attention_mask_16 = None
        transpose_61 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_61.contiguous()
        transpose_61 = None
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
        float_183 = attn_output_59.float()
        pow_88 = float_183.pow(2)
        mean_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_232 = mean_87 + 1e-06
        mean_87 = None
        rsqrt_87 = torch.rsqrt(add_232)
        add_232 = None
        output_174 = float_183 * rsqrt_87
        float_183 = rsqrt_87 = None
        float_184 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_233 = 1.0 + float_184
        float_184 = None
        output_175 = output_174 * add_233
        output_174 = add_233 = None
        hidden_states_115 = output_175.type_as(attn_output_59)
        output_175 = attn_output_59 = None
        hidden_states_116 = hidden_states_111 + hidden_states_115
        hidden_states_111 = hidden_states_115 = None
        float_185 = hidden_states_116.float()
        pow_89 = float_185.pow(2)
        mean_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_235 = mean_88 + 1e-06
        mean_88 = None
        rsqrt_88 = torch.rsqrt(add_235)
        add_235 = None
        output_176 = float_185 * rsqrt_88
        float_185 = rsqrt_88 = None
        float_186 = (
            l_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_236 = 1.0 + float_186
        float_186 = None
        output_177 = output_176 * add_236
        output_176 = add_236 = None
        hidden_states_117 = output_177.type_as(hidden_states_116)
        output_177 = None
        linear_102 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_14 = torch._C._nn.gelu(linear_102, approximate="tanh")
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_117 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_257 = gelu_14 * linear_103
        gelu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_257,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_257 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_187 = down_proj_14.float()
        pow_90 = float_187.pow(2)
        mean_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_237 = mean_89 + 1e-06
        mean_89 = None
        rsqrt_89 = torch.rsqrt(add_237)
        add_237 = None
        output_178 = float_187 * rsqrt_89
        float_187 = rsqrt_89 = None
        float_188 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_238 = 1.0 + float_188
        float_188 = None
        output_179 = output_178 * add_238
        output_178 = add_238 = None
        hidden_states_118 = output_179.type_as(down_proj_14)
        output_179 = down_proj_14 = None
        hidden_states_119 = hidden_states_116 + hidden_states_118
        hidden_states_116 = hidden_states_118 = None
        float_189 = hidden_states_119.float()
        pow_91 = float_189.pow(2)
        mean_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        add_240 = mean_90 + 1e-06
        mean_90 = None
        rsqrt_90 = torch.rsqrt(add_240)
        add_240 = None
        output_180 = float_189 * rsqrt_90
        float_189 = rsqrt_90 = None
        float_190 = (
            l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_241 = 1.0 + float_190
        float_190 = None
        output_181 = output_180 * add_241
        output_180 = add_241 = None
        hidden_states_120 = output_181.type_as(hidden_states_119)
        output_181 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_45 = linear_105.view((1, 20, -1, 256))
        linear_105 = None
        query_states_30 = view_45.transpose(1, 2)
        view_45 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_106.view((1, 20, -1, 256))
        linear_106 = None
        key_states_30 = view_46.transpose(1, 2)
        view_46 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_120 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_47 = linear_107.view((1, 20, -1, 256))
        linear_107 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        float_191 = query_states_30.float()
        pow_92 = float_191.pow(2)
        mean_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_242 = mean_91 + 1e-06
        mean_91 = None
        rsqrt_91 = torch.rsqrt(add_242)
        add_242 = None
        output_182 = float_191 * rsqrt_91
        float_191 = rsqrt_91 = None
        float_192 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_243 = 1.0 + float_192
        float_192 = None
        output_183 = output_182 * add_243
        output_182 = add_243 = None
        query_states_31 = output_183.type_as(query_states_30)
        output_183 = query_states_30 = None
        float_193 = key_states_30.float()
        pow_93 = float_193.pow(2)
        mean_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_244 = mean_92 + 1e-06
        mean_92 = None
        rsqrt_92 = torch.rsqrt(add_244)
        add_244 = None
        output_184 = float_193 * rsqrt_92
        float_193 = rsqrt_92 = None
        float_194 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_245 = 1.0 + float_194
        float_194 = None
        output_185 = output_184 * add_245
        output_184 = add_245 = None
        key_states_31 = output_185.type_as(key_states_30)
        output_185 = key_states_30 = None
        cos_21 = cos_4.unsqueeze(1)
        sin_21 = sin_4.unsqueeze(1)
        mul_266 = query_states_31 * cos_21
        x1_30 = query_states_31[(Ellipsis, slice(None, 128, None))]
        x2_30 = query_states_31[(Ellipsis, slice(128, None, None))]
        query_states_31 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_32 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_267 = cat_32 * sin_21
        cat_32 = None
        q_embed_15 = mul_266 + mul_267
        mul_266 = mul_267 = None
        mul_268 = key_states_31 * cos_21
        cos_21 = None
        x1_31 = key_states_31[(Ellipsis, slice(None, 128, None))]
        x2_31 = key_states_31[(Ellipsis, slice(128, None, None))]
        key_states_31 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_33 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_269 = cat_33 * sin_21
        cat_33 = sin_21 = None
        k_embed_15 = mul_268 + mul_269
        mul_268 = mul_269 = None
        getitem_113 = k_embed_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_15 = None
        hidden_states_121 = getitem_113.expand(1, 1, 4, 20, 256)
        getitem_113 = None
        key_30 = hidden_states_121.reshape(1, 4, 20, 256)
        hidden_states_121 = None
        getitem_114 = value_states_15[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_15 = None
        hidden_states_122 = getitem_114.expand(1, 1, 4, 20, 256)
        getitem_114 = None
        value_30 = hidden_states_122.reshape(1, 4, 20, 256)
        hidden_states_122 = None
        attention_mask_17 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_15 = key_31 = value_31 = attention_mask_17 = None
        transpose_65 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_65.contiguous()
        transpose_65 = None
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
        float_195 = attn_output_63.float()
        pow_94 = float_195.pow(2)
        mean_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_248 = mean_93 + 1e-06
        mean_93 = None
        rsqrt_93 = torch.rsqrt(add_248)
        add_248 = None
        output_186 = float_195 * rsqrt_93
        float_195 = rsqrt_93 = None
        float_196 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_249 = 1.0 + float_196
        float_196 = None
        output_187 = output_186 * add_249
        output_186 = add_249 = None
        hidden_states_123 = output_187.type_as(attn_output_63)
        output_187 = attn_output_63 = None
        hidden_states_124 = hidden_states_119 + hidden_states_123
        hidden_states_119 = hidden_states_123 = None
        float_197 = hidden_states_124.float()
        pow_95 = float_197.pow(2)
        mean_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        add_251 = mean_94 + 1e-06
        mean_94 = None
        rsqrt_94 = torch.rsqrt(add_251)
        add_251 = None
        output_188 = float_197 * rsqrt_94
        float_197 = rsqrt_94 = None
        float_198 = (
            l_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_252 = 1.0 + float_198
        float_198 = None
        output_189 = output_188 * add_252
        output_188 = add_252 = None
        hidden_states_125 = output_189.type_as(hidden_states_124)
        output_189 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_15 = torch._C._nn.gelu(linear_109, approximate="tanh")
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_125 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_274 = gelu_15 * linear_110
        gelu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_274,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_274 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_199 = down_proj_15.float()
        pow_96 = float_199.pow(2)
        mean_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_253 = mean_95 + 1e-06
        mean_95 = None
        rsqrt_95 = torch.rsqrt(add_253)
        add_253 = None
        output_190 = float_199 * rsqrt_95
        float_199 = rsqrt_95 = None
        float_200 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_254 = 1.0 + float_200
        float_200 = None
        output_191 = output_190 * add_254
        output_190 = add_254 = None
        hidden_states_126 = output_191.type_as(down_proj_15)
        output_191 = down_proj_15 = None
        hidden_states_127 = hidden_states_124 + hidden_states_126
        hidden_states_124 = hidden_states_126 = None
        float_201 = hidden_states_127.float()
        pow_97 = float_201.pow(2)
        mean_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_256 = mean_96 + 1e-06
        mean_96 = None
        rsqrt_96 = torch.rsqrt(add_256)
        add_256 = None
        output_192 = float_201 * rsqrt_96
        float_201 = rsqrt_96 = None
        float_202 = (
            l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_257 = 1.0 + float_202
        float_202 = None
        output_193 = output_192 * add_257
        output_192 = add_257 = None
        hidden_states_128 = output_193.type_as(hidden_states_127)
        output_193 = None
        linear_112 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_48 = linear_112.view((1, 20, -1, 256))
        linear_112 = None
        query_states_32 = view_48.transpose(1, 2)
        view_48 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_113.view((1, 20, -1, 256))
        linear_113 = None
        key_states_32 = view_49.transpose(1, 2)
        view_49 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_128,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_128 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_50 = linear_114.view((1, 20, -1, 256))
        linear_114 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        float_203 = query_states_32.float()
        pow_98 = float_203.pow(2)
        mean_97 = pow_98.mean(-1, keepdim=True)
        pow_98 = None
        add_258 = mean_97 + 1e-06
        mean_97 = None
        rsqrt_97 = torch.rsqrt(add_258)
        add_258 = None
        output_194 = float_203 * rsqrt_97
        float_203 = rsqrt_97 = None
        float_204 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_259 = 1.0 + float_204
        float_204 = None
        output_195 = output_194 * add_259
        output_194 = add_259 = None
        query_states_33 = output_195.type_as(query_states_32)
        output_195 = query_states_32 = None
        float_205 = key_states_32.float()
        pow_99 = float_205.pow(2)
        mean_98 = pow_99.mean(-1, keepdim=True)
        pow_99 = None
        add_260 = mean_98 + 1e-06
        mean_98 = None
        rsqrt_98 = torch.rsqrt(add_260)
        add_260 = None
        output_196 = float_205 * rsqrt_98
        float_205 = rsqrt_98 = None
        float_206 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_261 = 1.0 + float_206
        float_206 = None
        output_197 = output_196 * add_261
        output_196 = add_261 = None
        key_states_33 = output_197.type_as(key_states_32)
        output_197 = key_states_32 = None
        cos_22 = cos_4.unsqueeze(1)
        sin_22 = sin_4.unsqueeze(1)
        mul_283 = query_states_33 * cos_22
        x1_32 = query_states_33[(Ellipsis, slice(None, 128, None))]
        x2_32 = query_states_33[(Ellipsis, slice(128, None, None))]
        query_states_33 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_34 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_284 = cat_34 * sin_22
        cat_34 = None
        q_embed_16 = mul_283 + mul_284
        mul_283 = mul_284 = None
        mul_285 = key_states_33 * cos_22
        cos_22 = None
        x1_33 = key_states_33[(Ellipsis, slice(None, 128, None))]
        x2_33 = key_states_33[(Ellipsis, slice(128, None, None))]
        key_states_33 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_35 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_286 = cat_35 * sin_22
        cat_35 = sin_22 = None
        k_embed_16 = mul_285 + mul_286
        mul_285 = mul_286 = None
        getitem_120 = k_embed_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_16 = None
        hidden_states_129 = getitem_120.expand(1, 1, 4, 20, 256)
        getitem_120 = None
        key_32 = hidden_states_129.reshape(1, 4, 20, 256)
        hidden_states_129 = None
        getitem_121 = value_states_16[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_16 = None
        hidden_states_130 = getitem_121.expand(1, 1, 4, 20, 256)
        getitem_121 = None
        value_32 = hidden_states_130.reshape(1, 4, 20, 256)
        hidden_states_130 = None
        attention_mask_18 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_16 = key_33 = value_33 = attention_mask_18 = None
        transpose_69 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_69.contiguous()
        transpose_69 = None
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
        float_207 = attn_output_67.float()
        pow_100 = float_207.pow(2)
        mean_99 = pow_100.mean(-1, keepdim=True)
        pow_100 = None
        add_264 = mean_99 + 1e-06
        mean_99 = None
        rsqrt_99 = torch.rsqrt(add_264)
        add_264 = None
        output_198 = float_207 * rsqrt_99
        float_207 = rsqrt_99 = None
        float_208 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_265 = 1.0 + float_208
        float_208 = None
        output_199 = output_198 * add_265
        output_198 = add_265 = None
        hidden_states_131 = output_199.type_as(attn_output_67)
        output_199 = attn_output_67 = None
        hidden_states_132 = hidden_states_127 + hidden_states_131
        hidden_states_127 = hidden_states_131 = None
        float_209 = hidden_states_132.float()
        pow_101 = float_209.pow(2)
        mean_100 = pow_101.mean(-1, keepdim=True)
        pow_101 = None
        add_267 = mean_100 + 1e-06
        mean_100 = None
        rsqrt_100 = torch.rsqrt(add_267)
        add_267 = None
        output_200 = float_209 * rsqrt_100
        float_209 = rsqrt_100 = None
        float_210 = (
            l_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_268 = 1.0 + float_210
        float_210 = None
        output_201 = output_200 * add_268
        output_200 = add_268 = None
        hidden_states_133 = output_201.type_as(hidden_states_132)
        output_201 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_16 = torch._C._nn.gelu(linear_116, approximate="tanh")
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_133,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_133 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_291 = gelu_16 * linear_117
        gelu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_291,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_291 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_211 = down_proj_16.float()
        pow_102 = float_211.pow(2)
        mean_101 = pow_102.mean(-1, keepdim=True)
        pow_102 = None
        add_269 = mean_101 + 1e-06
        mean_101 = None
        rsqrt_101 = torch.rsqrt(add_269)
        add_269 = None
        output_202 = float_211 * rsqrt_101
        float_211 = rsqrt_101 = None
        float_212 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_270 = 1.0 + float_212
        float_212 = None
        output_203 = output_202 * add_270
        output_202 = add_270 = None
        hidden_states_134 = output_203.type_as(down_proj_16)
        output_203 = down_proj_16 = None
        hidden_states_135 = hidden_states_132 + hidden_states_134
        hidden_states_132 = hidden_states_134 = None
        float_213 = hidden_states_135.float()
        pow_103 = float_213.pow(2)
        mean_102 = pow_103.mean(-1, keepdim=True)
        pow_103 = None
        add_272 = mean_102 + 1e-06
        mean_102 = None
        rsqrt_102 = torch.rsqrt(add_272)
        add_272 = None
        output_204 = float_213 * rsqrt_102
        float_213 = rsqrt_102 = None
        float_214 = (
            l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_273 = 1.0 + float_214
        float_214 = None
        output_205 = output_204 * add_273
        output_204 = add_273 = None
        hidden_states_136 = output_205.type_as(hidden_states_135)
        output_205 = None
        linear_119 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_51 = linear_119.view((1, 20, -1, 256))
        linear_119 = None
        query_states_34 = view_51.transpose(1, 2)
        view_51 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_120.view((1, 20, -1, 256))
        linear_120 = None
        key_states_34 = view_52.transpose(1, 2)
        view_52 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_136,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_136 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_53 = linear_121.view((1, 20, -1, 256))
        linear_121 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        float_215 = query_states_34.float()
        pow_104 = float_215.pow(2)
        mean_103 = pow_104.mean(-1, keepdim=True)
        pow_104 = None
        add_274 = mean_103 + 1e-06
        mean_103 = None
        rsqrt_103 = torch.rsqrt(add_274)
        add_274 = None
        output_206 = float_215 * rsqrt_103
        float_215 = rsqrt_103 = None
        float_216 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_275 = 1.0 + float_216
        float_216 = None
        output_207 = output_206 * add_275
        output_206 = add_275 = None
        query_states_35 = output_207.type_as(query_states_34)
        output_207 = query_states_34 = None
        float_217 = key_states_34.float()
        pow_105 = float_217.pow(2)
        mean_104 = pow_105.mean(-1, keepdim=True)
        pow_105 = None
        add_276 = mean_104 + 1e-06
        mean_104 = None
        rsqrt_104 = torch.rsqrt(add_276)
        add_276 = None
        output_208 = float_217 * rsqrt_104
        float_217 = rsqrt_104 = None
        float_218 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_277 = 1.0 + float_218
        float_218 = None
        output_209 = output_208 * add_277
        output_208 = add_277 = None
        key_states_35 = output_209.type_as(key_states_34)
        output_209 = key_states_34 = None
        cos_23 = cos_10.unsqueeze(1)
        sin_23 = sin_10.unsqueeze(1)
        mul_300 = query_states_35 * cos_23
        x1_34 = query_states_35[(Ellipsis, slice(None, 128, None))]
        x2_34 = query_states_35[(Ellipsis, slice(128, None, None))]
        query_states_35 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_36 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_301 = cat_36 * sin_23
        cat_36 = None
        q_embed_17 = mul_300 + mul_301
        mul_300 = mul_301 = None
        mul_302 = key_states_35 * cos_23
        cos_23 = None
        x1_35 = key_states_35[(Ellipsis, slice(None, 128, None))]
        x2_35 = key_states_35[(Ellipsis, slice(128, None, None))]
        key_states_35 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_37 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_303 = cat_37 * sin_23
        cat_37 = sin_23 = None
        k_embed_17 = mul_302 + mul_303
        mul_302 = mul_303 = None
        getitem_127 = k_embed_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_17 = None
        hidden_states_137 = getitem_127.expand(1, 1, 4, 20, 256)
        getitem_127 = None
        key_34 = hidden_states_137.reshape(1, 4, 20, 256)
        hidden_states_137 = None
        getitem_128 = value_states_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_17 = None
        hidden_states_138 = getitem_128.expand(1, 1, 4, 20, 256)
        getitem_128 = None
        value_34 = hidden_states_138.reshape(1, 4, 20, 256)
        hidden_states_138 = None
        attention_mask_19 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_17 = key_35 = value_35 = attention_mask_19 = None
        transpose_73 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_73.contiguous()
        transpose_73 = None
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
        float_219 = attn_output_71.float()
        pow_106 = float_219.pow(2)
        mean_105 = pow_106.mean(-1, keepdim=True)
        pow_106 = None
        add_280 = mean_105 + 1e-06
        mean_105 = None
        rsqrt_105 = torch.rsqrt(add_280)
        add_280 = None
        output_210 = float_219 * rsqrt_105
        float_219 = rsqrt_105 = None
        float_220 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_281 = 1.0 + float_220
        float_220 = None
        output_211 = output_210 * add_281
        output_210 = add_281 = None
        hidden_states_139 = output_211.type_as(attn_output_71)
        output_211 = attn_output_71 = None
        hidden_states_140 = hidden_states_135 + hidden_states_139
        hidden_states_135 = hidden_states_139 = None
        float_221 = hidden_states_140.float()
        pow_107 = float_221.pow(2)
        mean_106 = pow_107.mean(-1, keepdim=True)
        pow_107 = None
        add_283 = mean_106 + 1e-06
        mean_106 = None
        rsqrt_106 = torch.rsqrt(add_283)
        add_283 = None
        output_212 = float_221 * rsqrt_106
        float_221 = rsqrt_106 = None
        float_222 = (
            l_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_284 = 1.0 + float_222
        float_222 = None
        output_213 = output_212 * add_284
        output_212 = add_284 = None
        hidden_states_141 = output_213.type_as(hidden_states_140)
        output_213 = None
        linear_123 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_17 = torch._C._nn.gelu(linear_123, approximate="tanh")
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_141,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_141 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_308 = gelu_17 * linear_124
        gelu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_308,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_308 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_223 = down_proj_17.float()
        pow_108 = float_223.pow(2)
        mean_107 = pow_108.mean(-1, keepdim=True)
        pow_108 = None
        add_285 = mean_107 + 1e-06
        mean_107 = None
        rsqrt_107 = torch.rsqrt(add_285)
        add_285 = None
        output_214 = float_223 * rsqrt_107
        float_223 = rsqrt_107 = None
        float_224 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_286 = 1.0 + float_224
        float_224 = None
        output_215 = output_214 * add_286
        output_214 = add_286 = None
        hidden_states_142 = output_215.type_as(down_proj_17)
        output_215 = down_proj_17 = None
        hidden_states_143 = hidden_states_140 + hidden_states_142
        hidden_states_140 = hidden_states_142 = None
        float_225 = hidden_states_143.float()
        pow_109 = float_225.pow(2)
        mean_108 = pow_109.mean(-1, keepdim=True)
        pow_109 = None
        add_288 = mean_108 + 1e-06
        mean_108 = None
        rsqrt_108 = torch.rsqrt(add_288)
        add_288 = None
        output_216 = float_225 * rsqrt_108
        float_225 = rsqrt_108 = None
        float_226 = (
            l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_289 = 1.0 + float_226
        float_226 = None
        output_217 = output_216 * add_289
        output_216 = add_289 = None
        hidden_states_144 = output_217.type_as(hidden_states_143)
        output_217 = None
        linear_126 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_54 = linear_126.view((1, 20, -1, 256))
        linear_126 = None
        query_states_36 = view_54.transpose(1, 2)
        view_54 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_55 = linear_127.view((1, 20, -1, 256))
        linear_127 = None
        key_states_36 = view_55.transpose(1, 2)
        view_55 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_144,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_144 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_56 = linear_128.view((1, 20, -1, 256))
        linear_128 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        float_227 = query_states_36.float()
        pow_110 = float_227.pow(2)
        mean_109 = pow_110.mean(-1, keepdim=True)
        pow_110 = None
        add_290 = mean_109 + 1e-06
        mean_109 = None
        rsqrt_109 = torch.rsqrt(add_290)
        add_290 = None
        output_218 = float_227 * rsqrt_109
        float_227 = rsqrt_109 = None
        float_228 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_291 = 1.0 + float_228
        float_228 = None
        output_219 = output_218 * add_291
        output_218 = add_291 = None
        query_states_37 = output_219.type_as(query_states_36)
        output_219 = query_states_36 = None
        float_229 = key_states_36.float()
        pow_111 = float_229.pow(2)
        mean_110 = pow_111.mean(-1, keepdim=True)
        pow_111 = None
        add_292 = mean_110 + 1e-06
        mean_110 = None
        rsqrt_110 = torch.rsqrt(add_292)
        add_292 = None
        output_220 = float_229 * rsqrt_110
        float_229 = rsqrt_110 = None
        float_230 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_293 = 1.0 + float_230
        float_230 = None
        output_221 = output_220 * add_293
        output_220 = add_293 = None
        key_states_37 = output_221.type_as(key_states_36)
        output_221 = key_states_36 = None
        cos_24 = cos_4.unsqueeze(1)
        sin_24 = sin_4.unsqueeze(1)
        mul_317 = query_states_37 * cos_24
        x1_36 = query_states_37[(Ellipsis, slice(None, 128, None))]
        x2_36 = query_states_37[(Ellipsis, slice(128, None, None))]
        query_states_37 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_38 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_318 = cat_38 * sin_24
        cat_38 = None
        q_embed_18 = mul_317 + mul_318
        mul_317 = mul_318 = None
        mul_319 = key_states_37 * cos_24
        cos_24 = None
        x1_37 = key_states_37[(Ellipsis, slice(None, 128, None))]
        x2_37 = key_states_37[(Ellipsis, slice(128, None, None))]
        key_states_37 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_39 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_320 = cat_39 * sin_24
        cat_39 = sin_24 = None
        k_embed_18 = mul_319 + mul_320
        mul_319 = mul_320 = None
        getitem_134 = k_embed_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_18 = None
        hidden_states_145 = getitem_134.expand(1, 1, 4, 20, 256)
        getitem_134 = None
        key_36 = hidden_states_145.reshape(1, 4, 20, 256)
        hidden_states_145 = None
        getitem_135 = value_states_18[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_18 = None
        hidden_states_146 = getitem_135.expand(1, 1, 4, 20, 256)
        getitem_135 = None
        value_36 = hidden_states_146.reshape(1, 4, 20, 256)
        hidden_states_146 = None
        attention_mask_20 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_18 = key_37 = value_37 = attention_mask_20 = None
        transpose_77 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_77.contiguous()
        transpose_77 = None
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
        float_231 = attn_output_75.float()
        pow_112 = float_231.pow(2)
        mean_111 = pow_112.mean(-1, keepdim=True)
        pow_112 = None
        add_296 = mean_111 + 1e-06
        mean_111 = None
        rsqrt_111 = torch.rsqrt(add_296)
        add_296 = None
        output_222 = float_231 * rsqrt_111
        float_231 = rsqrt_111 = None
        float_232 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_297 = 1.0 + float_232
        float_232 = None
        output_223 = output_222 * add_297
        output_222 = add_297 = None
        hidden_states_147 = output_223.type_as(attn_output_75)
        output_223 = attn_output_75 = None
        hidden_states_148 = hidden_states_143 + hidden_states_147
        hidden_states_143 = hidden_states_147 = None
        float_233 = hidden_states_148.float()
        pow_113 = float_233.pow(2)
        mean_112 = pow_113.mean(-1, keepdim=True)
        pow_113 = None
        add_299 = mean_112 + 1e-06
        mean_112 = None
        rsqrt_112 = torch.rsqrt(add_299)
        add_299 = None
        output_224 = float_233 * rsqrt_112
        float_233 = rsqrt_112 = None
        float_234 = (
            l_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_300 = 1.0 + float_234
        float_234 = None
        output_225 = output_224 * add_300
        output_224 = add_300 = None
        hidden_states_149 = output_225.type_as(hidden_states_148)
        output_225 = None
        linear_130 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_18 = torch._C._nn.gelu(linear_130, approximate="tanh")
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_149 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_325 = gelu_18 * linear_131
        gelu_18 = linear_131 = None
        down_proj_18 = torch._C._nn.linear(
            mul_325,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_325 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_235 = down_proj_18.float()
        pow_114 = float_235.pow(2)
        mean_113 = pow_114.mean(-1, keepdim=True)
        pow_114 = None
        add_301 = mean_113 + 1e-06
        mean_113 = None
        rsqrt_113 = torch.rsqrt(add_301)
        add_301 = None
        output_226 = float_235 * rsqrt_113
        float_235 = rsqrt_113 = None
        float_236 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_302 = 1.0 + float_236
        float_236 = None
        output_227 = output_226 * add_302
        output_226 = add_302 = None
        hidden_states_150 = output_227.type_as(down_proj_18)
        output_227 = down_proj_18 = None
        hidden_states_151 = hidden_states_148 + hidden_states_150
        hidden_states_148 = hidden_states_150 = None
        float_237 = hidden_states_151.float()
        pow_115 = float_237.pow(2)
        mean_114 = pow_115.mean(-1, keepdim=True)
        pow_115 = None
        add_304 = mean_114 + 1e-06
        mean_114 = None
        rsqrt_114 = torch.rsqrt(add_304)
        add_304 = None
        output_228 = float_237 * rsqrt_114
        float_237 = rsqrt_114 = None
        float_238 = (
            l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_305 = 1.0 + float_238
        float_238 = None
        output_229 = output_228 * add_305
        output_228 = add_305 = None
        hidden_states_152 = output_229.type_as(hidden_states_151)
        output_229 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_57 = linear_133.view((1, 20, -1, 256))
        linear_133 = None
        query_states_38 = view_57.transpose(1, 2)
        view_57 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_58 = linear_134.view((1, 20, -1, 256))
        linear_134 = None
        key_states_38 = view_58.transpose(1, 2)
        view_58 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_152 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_59 = linear_135.view((1, 20, -1, 256))
        linear_135 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        float_239 = query_states_38.float()
        pow_116 = float_239.pow(2)
        mean_115 = pow_116.mean(-1, keepdim=True)
        pow_116 = None
        add_306 = mean_115 + 1e-06
        mean_115 = None
        rsqrt_115 = torch.rsqrt(add_306)
        add_306 = None
        output_230 = float_239 * rsqrt_115
        float_239 = rsqrt_115 = None
        float_240 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_307 = 1.0 + float_240
        float_240 = None
        output_231 = output_230 * add_307
        output_230 = add_307 = None
        query_states_39 = output_231.type_as(query_states_38)
        output_231 = query_states_38 = None
        float_241 = key_states_38.float()
        pow_117 = float_241.pow(2)
        mean_116 = pow_117.mean(-1, keepdim=True)
        pow_117 = None
        add_308 = mean_116 + 1e-06
        mean_116 = None
        rsqrt_116 = torch.rsqrt(add_308)
        add_308 = None
        output_232 = float_241 * rsqrt_116
        float_241 = rsqrt_116 = None
        float_242 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_309 = 1.0 + float_242
        float_242 = None
        output_233 = output_232 * add_309
        output_232 = add_309 = None
        key_states_39 = output_233.type_as(key_states_38)
        output_233 = key_states_38 = None
        cos_25 = cos_4.unsqueeze(1)
        sin_25 = sin_4.unsqueeze(1)
        mul_334 = query_states_39 * cos_25
        x1_38 = query_states_39[(Ellipsis, slice(None, 128, None))]
        x2_38 = query_states_39[(Ellipsis, slice(128, None, None))]
        query_states_39 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_40 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_335 = cat_40 * sin_25
        cat_40 = None
        q_embed_19 = mul_334 + mul_335
        mul_334 = mul_335 = None
        mul_336 = key_states_39 * cos_25
        cos_25 = None
        x1_39 = key_states_39[(Ellipsis, slice(None, 128, None))]
        x2_39 = key_states_39[(Ellipsis, slice(128, None, None))]
        key_states_39 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_41 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_337 = cat_41 * sin_25
        cat_41 = sin_25 = None
        k_embed_19 = mul_336 + mul_337
        mul_336 = mul_337 = None
        getitem_141 = k_embed_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_19 = None
        hidden_states_153 = getitem_141.expand(1, 1, 4, 20, 256)
        getitem_141 = None
        key_38 = hidden_states_153.reshape(1, 4, 20, 256)
        hidden_states_153 = None
        getitem_142 = value_states_19[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_19 = None
        hidden_states_154 = getitem_142.expand(1, 1, 4, 20, 256)
        getitem_142 = None
        value_38 = hidden_states_154.reshape(1, 4, 20, 256)
        hidden_states_154 = None
        attention_mask_21 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_19 = key_39 = value_39 = attention_mask_21 = None
        transpose_81 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_81.contiguous()
        transpose_81 = None
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
        float_243 = attn_output_79.float()
        pow_118 = float_243.pow(2)
        mean_117 = pow_118.mean(-1, keepdim=True)
        pow_118 = None
        add_312 = mean_117 + 1e-06
        mean_117 = None
        rsqrt_117 = torch.rsqrt(add_312)
        add_312 = None
        output_234 = float_243 * rsqrt_117
        float_243 = rsqrt_117 = None
        float_244 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_313 = 1.0 + float_244
        float_244 = None
        output_235 = output_234 * add_313
        output_234 = add_313 = None
        hidden_states_155 = output_235.type_as(attn_output_79)
        output_235 = attn_output_79 = None
        hidden_states_156 = hidden_states_151 + hidden_states_155
        hidden_states_151 = hidden_states_155 = None
        float_245 = hidden_states_156.float()
        pow_119 = float_245.pow(2)
        mean_118 = pow_119.mean(-1, keepdim=True)
        pow_119 = None
        add_315 = mean_118 + 1e-06
        mean_118 = None
        rsqrt_118 = torch.rsqrt(add_315)
        add_315 = None
        output_236 = float_245 * rsqrt_118
        float_245 = rsqrt_118 = None
        float_246 = (
            l_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_316 = 1.0 + float_246
        float_246 = None
        output_237 = output_236 * add_316
        output_236 = add_316 = None
        hidden_states_157 = output_237.type_as(hidden_states_156)
        output_237 = None
        linear_137 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_19 = torch._C._nn.gelu(linear_137, approximate="tanh")
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_157 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_342 = gelu_19 * linear_138
        gelu_19 = linear_138 = None
        down_proj_19 = torch._C._nn.linear(
            mul_342,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_342 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_247 = down_proj_19.float()
        pow_120 = float_247.pow(2)
        mean_119 = pow_120.mean(-1, keepdim=True)
        pow_120 = None
        add_317 = mean_119 + 1e-06
        mean_119 = None
        rsqrt_119 = torch.rsqrt(add_317)
        add_317 = None
        output_238 = float_247 * rsqrt_119
        float_247 = rsqrt_119 = None
        float_248 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_318 = 1.0 + float_248
        float_248 = None
        output_239 = output_238 * add_318
        output_238 = add_318 = None
        hidden_states_158 = output_239.type_as(down_proj_19)
        output_239 = down_proj_19 = None
        hidden_states_159 = hidden_states_156 + hidden_states_158
        hidden_states_156 = hidden_states_158 = None
        float_249 = hidden_states_159.float()
        pow_121 = float_249.pow(2)
        mean_120 = pow_121.mean(-1, keepdim=True)
        pow_121 = None
        add_320 = mean_120 + 1e-06
        mean_120 = None
        rsqrt_120 = torch.rsqrt(add_320)
        add_320 = None
        output_240 = float_249 * rsqrt_120
        float_249 = rsqrt_120 = None
        float_250 = (
            l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_321 = 1.0 + float_250
        float_250 = None
        output_241 = output_240 * add_321
        output_240 = add_321 = None
        hidden_states_160 = output_241.type_as(hidden_states_159)
        output_241 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_60 = linear_140.view((1, 20, -1, 256))
        linear_140 = None
        query_states_40 = view_60.transpose(1, 2)
        view_60 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_61 = linear_141.view((1, 20, -1, 256))
        linear_141 = None
        key_states_40 = view_61.transpose(1, 2)
        view_61 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_160,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_160 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_62 = linear_142.view((1, 20, -1, 256))
        linear_142 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        float_251 = query_states_40.float()
        pow_122 = float_251.pow(2)
        mean_121 = pow_122.mean(-1, keepdim=True)
        pow_122 = None
        add_322 = mean_121 + 1e-06
        mean_121 = None
        rsqrt_121 = torch.rsqrt(add_322)
        add_322 = None
        output_242 = float_251 * rsqrt_121
        float_251 = rsqrt_121 = None
        float_252 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_323 = 1.0 + float_252
        float_252 = None
        output_243 = output_242 * add_323
        output_242 = add_323 = None
        query_states_41 = output_243.type_as(query_states_40)
        output_243 = query_states_40 = None
        float_253 = key_states_40.float()
        pow_123 = float_253.pow(2)
        mean_122 = pow_123.mean(-1, keepdim=True)
        pow_123 = None
        add_324 = mean_122 + 1e-06
        mean_122 = None
        rsqrt_122 = torch.rsqrt(add_324)
        add_324 = None
        output_244 = float_253 * rsqrt_122
        float_253 = rsqrt_122 = None
        float_254 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_325 = 1.0 + float_254
        float_254 = None
        output_245 = output_244 * add_325
        output_244 = add_325 = None
        key_states_41 = output_245.type_as(key_states_40)
        output_245 = key_states_40 = None
        cos_26 = cos_4.unsqueeze(1)
        sin_26 = sin_4.unsqueeze(1)
        mul_351 = query_states_41 * cos_26
        x1_40 = query_states_41[(Ellipsis, slice(None, 128, None))]
        x2_40 = query_states_41[(Ellipsis, slice(128, None, None))]
        query_states_41 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_42 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_352 = cat_42 * sin_26
        cat_42 = None
        q_embed_20 = mul_351 + mul_352
        mul_351 = mul_352 = None
        mul_353 = key_states_41 * cos_26
        cos_26 = None
        x1_41 = key_states_41[(Ellipsis, slice(None, 128, None))]
        x2_41 = key_states_41[(Ellipsis, slice(128, None, None))]
        key_states_41 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_43 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_354 = cat_43 * sin_26
        cat_43 = sin_26 = None
        k_embed_20 = mul_353 + mul_354
        mul_353 = mul_354 = None
        getitem_148 = k_embed_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_20 = None
        hidden_states_161 = getitem_148.expand(1, 1, 4, 20, 256)
        getitem_148 = None
        key_40 = hidden_states_161.reshape(1, 4, 20, 256)
        hidden_states_161 = None
        getitem_149 = value_states_20[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_20 = None
        hidden_states_162 = getitem_149.expand(1, 1, 4, 20, 256)
        getitem_149 = None
        value_40 = hidden_states_162.reshape(1, 4, 20, 256)
        hidden_states_162 = None
        attention_mask_22 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_20 = key_41 = value_41 = attention_mask_22 = None
        transpose_85 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_85.contiguous()
        transpose_85 = None
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
        float_255 = attn_output_83.float()
        pow_124 = float_255.pow(2)
        mean_123 = pow_124.mean(-1, keepdim=True)
        pow_124 = None
        add_328 = mean_123 + 1e-06
        mean_123 = None
        rsqrt_123 = torch.rsqrt(add_328)
        add_328 = None
        output_246 = float_255 * rsqrt_123
        float_255 = rsqrt_123 = None
        float_256 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_329 = 1.0 + float_256
        float_256 = None
        output_247 = output_246 * add_329
        output_246 = add_329 = None
        hidden_states_163 = output_247.type_as(attn_output_83)
        output_247 = attn_output_83 = None
        hidden_states_164 = hidden_states_159 + hidden_states_163
        hidden_states_159 = hidden_states_163 = None
        float_257 = hidden_states_164.float()
        pow_125 = float_257.pow(2)
        mean_124 = pow_125.mean(-1, keepdim=True)
        pow_125 = None
        add_331 = mean_124 + 1e-06
        mean_124 = None
        rsqrt_124 = torch.rsqrt(add_331)
        add_331 = None
        output_248 = float_257 * rsqrt_124
        float_257 = rsqrt_124 = None
        float_258 = (
            l_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_332 = 1.0 + float_258
        float_258 = None
        output_249 = output_248 * add_332
        output_248 = add_332 = None
        hidden_states_165 = output_249.type_as(hidden_states_164)
        output_249 = None
        linear_144 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_20 = torch._C._nn.gelu(linear_144, approximate="tanh")
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_165,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_165 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_359 = gelu_20 * linear_145
        gelu_20 = linear_145 = None
        down_proj_20 = torch._C._nn.linear(
            mul_359,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_359 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_259 = down_proj_20.float()
        pow_126 = float_259.pow(2)
        mean_125 = pow_126.mean(-1, keepdim=True)
        pow_126 = None
        add_333 = mean_125 + 1e-06
        mean_125 = None
        rsqrt_125 = torch.rsqrt(add_333)
        add_333 = None
        output_250 = float_259 * rsqrt_125
        float_259 = rsqrt_125 = None
        float_260 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_334 = 1.0 + float_260
        float_260 = None
        output_251 = output_250 * add_334
        output_250 = add_334 = None
        hidden_states_166 = output_251.type_as(down_proj_20)
        output_251 = down_proj_20 = None
        hidden_states_167 = hidden_states_164 + hidden_states_166
        hidden_states_164 = hidden_states_166 = None
        float_261 = hidden_states_167.float()
        pow_127 = float_261.pow(2)
        mean_126 = pow_127.mean(-1, keepdim=True)
        pow_127 = None
        add_336 = mean_126 + 1e-06
        mean_126 = None
        rsqrt_126 = torch.rsqrt(add_336)
        add_336 = None
        output_252 = float_261 * rsqrt_126
        float_261 = rsqrt_126 = None
        float_262 = (
            l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_337 = 1.0 + float_262
        float_262 = None
        output_253 = output_252 * add_337
        output_252 = add_337 = None
        hidden_states_168 = output_253.type_as(hidden_states_167)
        output_253 = None
        linear_147 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_63 = linear_147.view((1, 20, -1, 256))
        linear_147 = None
        query_states_42 = view_63.transpose(1, 2)
        view_63 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_64 = linear_148.view((1, 20, -1, 256))
        linear_148 = None
        key_states_42 = view_64.transpose(1, 2)
        view_64 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_168,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_168 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_65 = linear_149.view((1, 20, -1, 256))
        linear_149 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        float_263 = query_states_42.float()
        pow_128 = float_263.pow(2)
        mean_127 = pow_128.mean(-1, keepdim=True)
        pow_128 = None
        add_338 = mean_127 + 1e-06
        mean_127 = None
        rsqrt_127 = torch.rsqrt(add_338)
        add_338 = None
        output_254 = float_263 * rsqrt_127
        float_263 = rsqrt_127 = None
        float_264 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_339 = 1.0 + float_264
        float_264 = None
        output_255 = output_254 * add_339
        output_254 = add_339 = None
        query_states_43 = output_255.type_as(query_states_42)
        output_255 = query_states_42 = None
        float_265 = key_states_42.float()
        pow_129 = float_265.pow(2)
        mean_128 = pow_129.mean(-1, keepdim=True)
        pow_129 = None
        add_340 = mean_128 + 1e-06
        mean_128 = None
        rsqrt_128 = torch.rsqrt(add_340)
        add_340 = None
        output_256 = float_265 * rsqrt_128
        float_265 = rsqrt_128 = None
        float_266 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_341 = 1.0 + float_266
        float_266 = None
        output_257 = output_256 * add_341
        output_256 = add_341 = None
        key_states_43 = output_257.type_as(key_states_42)
        output_257 = key_states_42 = None
        cos_27 = cos_4.unsqueeze(1)
        sin_27 = sin_4.unsqueeze(1)
        mul_368 = query_states_43 * cos_27
        x1_42 = query_states_43[(Ellipsis, slice(None, 128, None))]
        x2_42 = query_states_43[(Ellipsis, slice(128, None, None))]
        query_states_43 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_44 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_369 = cat_44 * sin_27
        cat_44 = None
        q_embed_21 = mul_368 + mul_369
        mul_368 = mul_369 = None
        mul_370 = key_states_43 * cos_27
        cos_27 = None
        x1_43 = key_states_43[(Ellipsis, slice(None, 128, None))]
        x2_43 = key_states_43[(Ellipsis, slice(128, None, None))]
        key_states_43 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_45 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_371 = cat_45 * sin_27
        cat_45 = sin_27 = None
        k_embed_21 = mul_370 + mul_371
        mul_370 = mul_371 = None
        getitem_155 = k_embed_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_21 = None
        hidden_states_169 = getitem_155.expand(1, 1, 4, 20, 256)
        getitem_155 = None
        key_42 = hidden_states_169.reshape(1, 4, 20, 256)
        hidden_states_169 = None
        getitem_156 = value_states_21[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_21 = None
        hidden_states_170 = getitem_156.expand(1, 1, 4, 20, 256)
        getitem_156 = None
        value_42 = hidden_states_170.reshape(1, 4, 20, 256)
        hidden_states_170 = None
        attention_mask_23 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_21 = key_43 = value_43 = attention_mask_23 = None
        transpose_89 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_89.contiguous()
        transpose_89 = None
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
        float_267 = attn_output_87.float()
        pow_130 = float_267.pow(2)
        mean_129 = pow_130.mean(-1, keepdim=True)
        pow_130 = None
        add_344 = mean_129 + 1e-06
        mean_129 = None
        rsqrt_129 = torch.rsqrt(add_344)
        add_344 = None
        output_258 = float_267 * rsqrt_129
        float_267 = rsqrt_129 = None
        float_268 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_345 = 1.0 + float_268
        float_268 = None
        output_259 = output_258 * add_345
        output_258 = add_345 = None
        hidden_states_171 = output_259.type_as(attn_output_87)
        output_259 = attn_output_87 = None
        hidden_states_172 = hidden_states_167 + hidden_states_171
        hidden_states_167 = hidden_states_171 = None
        float_269 = hidden_states_172.float()
        pow_131 = float_269.pow(2)
        mean_130 = pow_131.mean(-1, keepdim=True)
        pow_131 = None
        add_347 = mean_130 + 1e-06
        mean_130 = None
        rsqrt_130 = torch.rsqrt(add_347)
        add_347 = None
        output_260 = float_269 * rsqrt_130
        float_269 = rsqrt_130 = None
        float_270 = (
            l_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_348 = 1.0 + float_270
        float_270 = None
        output_261 = output_260 * add_348
        output_260 = add_348 = None
        hidden_states_173 = output_261.type_as(hidden_states_172)
        output_261 = None
        linear_151 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_21 = torch._C._nn.gelu(linear_151, approximate="tanh")
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_173,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_173 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_376 = gelu_21 * linear_152
        gelu_21 = linear_152 = None
        down_proj_21 = torch._C._nn.linear(
            mul_376,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_376 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_271 = down_proj_21.float()
        pow_132 = float_271.pow(2)
        mean_131 = pow_132.mean(-1, keepdim=True)
        pow_132 = None
        add_349 = mean_131 + 1e-06
        mean_131 = None
        rsqrt_131 = torch.rsqrt(add_349)
        add_349 = None
        output_262 = float_271 * rsqrt_131
        float_271 = rsqrt_131 = None
        float_272 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_350 = 1.0 + float_272
        float_272 = None
        output_263 = output_262 * add_350
        output_262 = add_350 = None
        hidden_states_174 = output_263.type_as(down_proj_21)
        output_263 = down_proj_21 = None
        hidden_states_175 = hidden_states_172 + hidden_states_174
        hidden_states_172 = hidden_states_174 = None
        float_273 = hidden_states_175.float()
        pow_133 = float_273.pow(2)
        mean_132 = pow_133.mean(-1, keepdim=True)
        pow_133 = None
        add_352 = mean_132 + 1e-06
        mean_132 = None
        rsqrt_132 = torch.rsqrt(add_352)
        add_352 = None
        output_264 = float_273 * rsqrt_132
        float_273 = rsqrt_132 = None
        float_274 = (
            l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_353 = 1.0 + float_274
        float_274 = None
        output_265 = output_264 * add_353
        output_264 = add_353 = None
        hidden_states_176 = output_265.type_as(hidden_states_175)
        output_265 = None
        linear_154 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_66 = linear_154.view((1, 20, -1, 256))
        linear_154 = None
        query_states_44 = view_66.transpose(1, 2)
        view_66 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_67 = linear_155.view((1, 20, -1, 256))
        linear_155 = None
        key_states_44 = view_67.transpose(1, 2)
        view_67 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_176,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_176 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_68 = linear_156.view((1, 20, -1, 256))
        linear_156 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        float_275 = query_states_44.float()
        pow_134 = float_275.pow(2)
        mean_133 = pow_134.mean(-1, keepdim=True)
        pow_134 = None
        add_354 = mean_133 + 1e-06
        mean_133 = None
        rsqrt_133 = torch.rsqrt(add_354)
        add_354 = None
        output_266 = float_275 * rsqrt_133
        float_275 = rsqrt_133 = None
        float_276 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_355 = 1.0 + float_276
        float_276 = None
        output_267 = output_266 * add_355
        output_266 = add_355 = None
        query_states_45 = output_267.type_as(query_states_44)
        output_267 = query_states_44 = None
        float_277 = key_states_44.float()
        pow_135 = float_277.pow(2)
        mean_134 = pow_135.mean(-1, keepdim=True)
        pow_135 = None
        add_356 = mean_134 + 1e-06
        mean_134 = None
        rsqrt_134 = torch.rsqrt(add_356)
        add_356 = None
        output_268 = float_277 * rsqrt_134
        float_277 = rsqrt_134 = None
        float_278 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_357 = 1.0 + float_278
        float_278 = None
        output_269 = output_268 * add_357
        output_268 = add_357 = None
        key_states_45 = output_269.type_as(key_states_44)
        output_269 = key_states_44 = None
        cos_28 = cos_4.unsqueeze(1)
        sin_28 = sin_4.unsqueeze(1)
        mul_385 = query_states_45 * cos_28
        x1_44 = query_states_45[(Ellipsis, slice(None, 128, None))]
        x2_44 = query_states_45[(Ellipsis, slice(128, None, None))]
        query_states_45 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_46 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_386 = cat_46 * sin_28
        cat_46 = None
        q_embed_22 = mul_385 + mul_386
        mul_385 = mul_386 = None
        mul_387 = key_states_45 * cos_28
        cos_28 = None
        x1_45 = key_states_45[(Ellipsis, slice(None, 128, None))]
        x2_45 = key_states_45[(Ellipsis, slice(128, None, None))]
        key_states_45 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_47 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_388 = cat_47 * sin_28
        cat_47 = sin_28 = None
        k_embed_22 = mul_387 + mul_388
        mul_387 = mul_388 = None
        getitem_162 = k_embed_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_22 = None
        hidden_states_177 = getitem_162.expand(1, 1, 4, 20, 256)
        getitem_162 = None
        key_44 = hidden_states_177.reshape(1, 4, 20, 256)
        hidden_states_177 = None
        getitem_163 = value_states_22[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_22 = None
        hidden_states_178 = getitem_163.expand(1, 1, 4, 20, 256)
        getitem_163 = None
        value_44 = hidden_states_178.reshape(1, 4, 20, 256)
        hidden_states_178 = None
        attention_mask_24 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_22 = key_45 = value_45 = attention_mask_24 = None
        transpose_93 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_93.contiguous()
        transpose_93 = None
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
        float_279 = attn_output_91.float()
        pow_136 = float_279.pow(2)
        mean_135 = pow_136.mean(-1, keepdim=True)
        pow_136 = None
        add_360 = mean_135 + 1e-06
        mean_135 = None
        rsqrt_135 = torch.rsqrt(add_360)
        add_360 = None
        output_270 = float_279 * rsqrt_135
        float_279 = rsqrt_135 = None
        float_280 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_361 = 1.0 + float_280
        float_280 = None
        output_271 = output_270 * add_361
        output_270 = add_361 = None
        hidden_states_179 = output_271.type_as(attn_output_91)
        output_271 = attn_output_91 = None
        hidden_states_180 = hidden_states_175 + hidden_states_179
        hidden_states_175 = hidden_states_179 = None
        float_281 = hidden_states_180.float()
        pow_137 = float_281.pow(2)
        mean_136 = pow_137.mean(-1, keepdim=True)
        pow_137 = None
        add_363 = mean_136 + 1e-06
        mean_136 = None
        rsqrt_136 = torch.rsqrt(add_363)
        add_363 = None
        output_272 = float_281 * rsqrt_136
        float_281 = rsqrt_136 = None
        float_282 = (
            l_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_364 = 1.0 + float_282
        float_282 = None
        output_273 = output_272 * add_364
        output_272 = add_364 = None
        hidden_states_181 = output_273.type_as(hidden_states_180)
        output_273 = None
        linear_158 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_22 = torch._C._nn.gelu(linear_158, approximate="tanh")
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_181 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_393 = gelu_22 * linear_159
        gelu_22 = linear_159 = None
        down_proj_22 = torch._C._nn.linear(
            mul_393,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_393 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_283 = down_proj_22.float()
        pow_138 = float_283.pow(2)
        mean_137 = pow_138.mean(-1, keepdim=True)
        pow_138 = None
        add_365 = mean_137 + 1e-06
        mean_137 = None
        rsqrt_137 = torch.rsqrt(add_365)
        add_365 = None
        output_274 = float_283 * rsqrt_137
        float_283 = rsqrt_137 = None
        float_284 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_366 = 1.0 + float_284
        float_284 = None
        output_275 = output_274 * add_366
        output_274 = add_366 = None
        hidden_states_182 = output_275.type_as(down_proj_22)
        output_275 = down_proj_22 = None
        hidden_states_183 = hidden_states_180 + hidden_states_182
        hidden_states_180 = hidden_states_182 = None
        float_285 = hidden_states_183.float()
        pow_139 = float_285.pow(2)
        mean_138 = pow_139.mean(-1, keepdim=True)
        pow_139 = None
        add_368 = mean_138 + 1e-06
        mean_138 = None
        rsqrt_138 = torch.rsqrt(add_368)
        add_368 = None
        output_276 = float_285 * rsqrt_138
        float_285 = rsqrt_138 = None
        float_286 = (
            l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_369 = 1.0 + float_286
        float_286 = None
        output_277 = output_276 * add_369
        output_276 = add_369 = None
        hidden_states_184 = output_277.type_as(hidden_states_183)
        output_277 = None
        linear_161 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_69 = linear_161.view((1, 20, -1, 256))
        linear_161 = None
        query_states_46 = view_69.transpose(1, 2)
        view_69 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_70 = linear_162.view((1, 20, -1, 256))
        linear_162 = None
        key_states_46 = view_70.transpose(1, 2)
        view_70 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_184,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_184 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_71 = linear_163.view((1, 20, -1, 256))
        linear_163 = None
        value_states_23 = view_71.transpose(1, 2)
        view_71 = None
        float_287 = query_states_46.float()
        pow_140 = float_287.pow(2)
        mean_139 = pow_140.mean(-1, keepdim=True)
        pow_140 = None
        add_370 = mean_139 + 1e-06
        mean_139 = None
        rsqrt_139 = torch.rsqrt(add_370)
        add_370 = None
        output_278 = float_287 * rsqrt_139
        float_287 = rsqrt_139 = None
        float_288 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_371 = 1.0 + float_288
        float_288 = None
        output_279 = output_278 * add_371
        output_278 = add_371 = None
        query_states_47 = output_279.type_as(query_states_46)
        output_279 = query_states_46 = None
        float_289 = key_states_46.float()
        pow_141 = float_289.pow(2)
        mean_140 = pow_141.mean(-1, keepdim=True)
        pow_141 = None
        add_372 = mean_140 + 1e-06
        mean_140 = None
        rsqrt_140 = torch.rsqrt(add_372)
        add_372 = None
        output_280 = float_289 * rsqrt_140
        float_289 = rsqrt_140 = None
        float_290 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_373 = 1.0 + float_290
        float_290 = None
        output_281 = output_280 * add_373
        output_280 = add_373 = None
        key_states_47 = output_281.type_as(key_states_46)
        output_281 = key_states_46 = None
        cos_29 = cos_10.unsqueeze(1)
        cos_10 = None
        sin_29 = sin_10.unsqueeze(1)
        sin_10 = None
        mul_402 = query_states_47 * cos_29
        x1_46 = query_states_47[(Ellipsis, slice(None, 128, None))]
        x2_46 = query_states_47[(Ellipsis, slice(128, None, None))]
        query_states_47 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_48 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_403 = cat_48 * sin_29
        cat_48 = None
        q_embed_23 = mul_402 + mul_403
        mul_402 = mul_403 = None
        mul_404 = key_states_47 * cos_29
        cos_29 = None
        x1_47 = key_states_47[(Ellipsis, slice(None, 128, None))]
        x2_47 = key_states_47[(Ellipsis, slice(128, None, None))]
        key_states_47 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_49 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_405 = cat_49 * sin_29
        cat_49 = sin_29 = None
        k_embed_23 = mul_404 + mul_405
        mul_404 = mul_405 = None
        getitem_169 = k_embed_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_23 = None
        hidden_states_185 = getitem_169.expand(1, 1, 4, 20, 256)
        getitem_169 = None
        key_46 = hidden_states_185.reshape(1, 4, 20, 256)
        hidden_states_185 = None
        getitem_170 = value_states_23[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_23 = None
        hidden_states_186 = getitem_170.expand(1, 1, 4, 20, 256)
        getitem_170 = None
        value_46 = hidden_states_186.reshape(1, 4, 20, 256)
        hidden_states_186 = None
        attention_mask_25 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        causal_mask = None
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
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_23 = key_47 = value_47 = attention_mask_25 = None
        transpose_97 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_97.contiguous()
        transpose_97 = None
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
        float_291 = attn_output_95.float()
        pow_142 = float_291.pow(2)
        mean_141 = pow_142.mean(-1, keepdim=True)
        pow_142 = None
        add_376 = mean_141 + 1e-06
        mean_141 = None
        rsqrt_141 = torch.rsqrt(add_376)
        add_376 = None
        output_282 = float_291 * rsqrt_141
        float_291 = rsqrt_141 = None
        float_292 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_377 = 1.0 + float_292
        float_292 = None
        output_283 = output_282 * add_377
        output_282 = add_377 = None
        hidden_states_187 = output_283.type_as(attn_output_95)
        output_283 = attn_output_95 = None
        hidden_states_188 = hidden_states_183 + hidden_states_187
        hidden_states_183 = hidden_states_187 = None
        float_293 = hidden_states_188.float()
        pow_143 = float_293.pow(2)
        mean_142 = pow_143.mean(-1, keepdim=True)
        pow_143 = None
        add_379 = mean_142 + 1e-06
        mean_142 = None
        rsqrt_142 = torch.rsqrt(add_379)
        add_379 = None
        output_284 = float_293 * rsqrt_142
        float_293 = rsqrt_142 = None
        float_294 = (
            l_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_380 = 1.0 + float_294
        float_294 = None
        output_285 = output_284 * add_380
        output_284 = add_380 = None
        hidden_states_189 = output_285.type_as(hidden_states_188)
        output_285 = None
        linear_165 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_23 = torch._C._nn.gelu(linear_165, approximate="tanh")
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_189,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_189 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_410 = gelu_23 * linear_166
        gelu_23 = linear_166 = None
        down_proj_23 = torch._C._nn.linear(
            mul_410,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_410 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_295 = down_proj_23.float()
        pow_144 = float_295.pow(2)
        mean_143 = pow_144.mean(-1, keepdim=True)
        pow_144 = None
        add_381 = mean_143 + 1e-06
        mean_143 = None
        rsqrt_143 = torch.rsqrt(add_381)
        add_381 = None
        output_286 = float_295 * rsqrt_143
        float_295 = rsqrt_143 = None
        float_296 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_382 = 1.0 + float_296
        float_296 = None
        output_287 = output_286 * add_382
        output_286 = add_382 = None
        hidden_states_190 = output_287.type_as(down_proj_23)
        output_287 = down_proj_23 = None
        hidden_states_191 = hidden_states_188 + hidden_states_190
        hidden_states_188 = hidden_states_190 = None
        float_297 = hidden_states_191.float()
        pow_145 = float_297.pow(2)
        mean_144 = pow_145.mean(-1, keepdim=True)
        pow_145 = None
        add_384 = mean_144 + 1e-06
        mean_144 = None
        rsqrt_144 = torch.rsqrt(add_384)
        add_384 = None
        output_288 = float_297 * rsqrt_144
        float_297 = rsqrt_144 = None
        float_298 = (
            l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_385 = 1.0 + float_298
        float_298 = None
        output_289 = output_288 * add_385
        output_288 = add_385 = None
        hidden_states_192 = output_289.type_as(hidden_states_191)
        output_289 = None
        linear_168 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_72 = linear_168.view((1, 20, -1, 256))
        linear_168 = None
        query_states_48 = view_72.transpose(1, 2)
        view_72 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_73 = linear_169.view((1, 20, -1, 256))
        linear_169 = None
        key_states_48 = view_73.transpose(1, 2)
        view_73 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_192,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_192 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_74 = linear_170.view((1, 20, -1, 256))
        linear_170 = None
        value_states_24 = view_74.transpose(1, 2)
        view_74 = None
        float_299 = query_states_48.float()
        pow_146 = float_299.pow(2)
        mean_145 = pow_146.mean(-1, keepdim=True)
        pow_146 = None
        add_386 = mean_145 + 1e-06
        mean_145 = None
        rsqrt_145 = torch.rsqrt(add_386)
        add_386 = None
        output_290 = float_299 * rsqrt_145
        float_299 = rsqrt_145 = None
        float_300 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_387 = 1.0 + float_300
        float_300 = None
        output_291 = output_290 * add_387
        output_290 = add_387 = None
        query_states_49 = output_291.type_as(query_states_48)
        output_291 = query_states_48 = None
        float_301 = key_states_48.float()
        pow_147 = float_301.pow(2)
        mean_146 = pow_147.mean(-1, keepdim=True)
        pow_147 = None
        add_388 = mean_146 + 1e-06
        mean_146 = None
        rsqrt_146 = torch.rsqrt(add_388)
        add_388 = None
        output_292 = float_301 * rsqrt_146
        float_301 = rsqrt_146 = None
        float_302 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_389 = 1.0 + float_302
        float_302 = None
        output_293 = output_292 * add_389
        output_292 = add_389 = None
        key_states_49 = output_293.type_as(key_states_48)
        output_293 = key_states_48 = None
        cos_30 = cos_4.unsqueeze(1)
        sin_30 = sin_4.unsqueeze(1)
        mul_419 = query_states_49 * cos_30
        x1_48 = query_states_49[(Ellipsis, slice(None, 128, None))]
        x2_48 = query_states_49[(Ellipsis, slice(128, None, None))]
        query_states_49 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_50 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_420 = cat_50 * sin_30
        cat_50 = None
        q_embed_24 = mul_419 + mul_420
        mul_419 = mul_420 = None
        mul_421 = key_states_49 * cos_30
        cos_30 = None
        x1_49 = key_states_49[(Ellipsis, slice(None, 128, None))]
        x2_49 = key_states_49[(Ellipsis, slice(128, None, None))]
        key_states_49 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_51 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_422 = cat_51 * sin_30
        cat_51 = sin_30 = None
        k_embed_24 = mul_421 + mul_422
        mul_421 = mul_422 = None
        getitem_176 = k_embed_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_24 = None
        hidden_states_193 = getitem_176.expand(1, 1, 4, 20, 256)
        getitem_176 = None
        key_48 = hidden_states_193.reshape(1, 4, 20, 256)
        hidden_states_193 = None
        getitem_177 = value_states_24[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_24 = None
        hidden_states_194 = getitem_177.expand(1, 1, 4, 20, 256)
        getitem_177 = None
        value_48 = hidden_states_194.reshape(1, 4, 20, 256)
        hidden_states_194 = None
        attention_mask_26 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
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
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_24 = key_49 = value_49 = attention_mask_26 = None
        transpose_101 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_101.contiguous()
        transpose_101 = None
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
        float_303 = attn_output_99.float()
        pow_148 = float_303.pow(2)
        mean_147 = pow_148.mean(-1, keepdim=True)
        pow_148 = None
        add_392 = mean_147 + 1e-06
        mean_147 = None
        rsqrt_147 = torch.rsqrt(add_392)
        add_392 = None
        output_294 = float_303 * rsqrt_147
        float_303 = rsqrt_147 = None
        float_304 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_393 = 1.0 + float_304
        float_304 = None
        output_295 = output_294 * add_393
        output_294 = add_393 = None
        hidden_states_195 = output_295.type_as(attn_output_99)
        output_295 = attn_output_99 = None
        hidden_states_196 = hidden_states_191 + hidden_states_195
        hidden_states_191 = hidden_states_195 = None
        float_305 = hidden_states_196.float()
        pow_149 = float_305.pow(2)
        mean_148 = pow_149.mean(-1, keepdim=True)
        pow_149 = None
        add_395 = mean_148 + 1e-06
        mean_148 = None
        rsqrt_148 = torch.rsqrt(add_395)
        add_395 = None
        output_296 = float_305 * rsqrt_148
        float_305 = rsqrt_148 = None
        float_306 = (
            l_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_396 = 1.0 + float_306
        float_306 = None
        output_297 = output_296 * add_396
        output_296 = add_396 = None
        hidden_states_197 = output_297.type_as(hidden_states_196)
        output_297 = None
        linear_172 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_24 = torch._C._nn.gelu(linear_172, approximate="tanh")
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_197,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_197 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_427 = gelu_24 * linear_173
        gelu_24 = linear_173 = None
        down_proj_24 = torch._C._nn.linear(
            mul_427,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_427 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_307 = down_proj_24.float()
        pow_150 = float_307.pow(2)
        mean_149 = pow_150.mean(-1, keepdim=True)
        pow_150 = None
        add_397 = mean_149 + 1e-06
        mean_149 = None
        rsqrt_149 = torch.rsqrt(add_397)
        add_397 = None
        output_298 = float_307 * rsqrt_149
        float_307 = rsqrt_149 = None
        float_308 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_398 = 1.0 + float_308
        float_308 = None
        output_299 = output_298 * add_398
        output_298 = add_398 = None
        hidden_states_198 = output_299.type_as(down_proj_24)
        output_299 = down_proj_24 = None
        hidden_states_199 = hidden_states_196 + hidden_states_198
        hidden_states_196 = hidden_states_198 = None
        float_309 = hidden_states_199.float()
        pow_151 = float_309.pow(2)
        mean_150 = pow_151.mean(-1, keepdim=True)
        pow_151 = None
        add_400 = mean_150 + 1e-06
        mean_150 = None
        rsqrt_150 = torch.rsqrt(add_400)
        add_400 = None
        output_300 = float_309 * rsqrt_150
        float_309 = rsqrt_150 = None
        float_310 = (
            l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = (
            None
        )
        add_401 = 1.0 + float_310
        float_310 = None
        output_301 = output_300 * add_401
        output_300 = add_401 = None
        hidden_states_200 = output_301.type_as(hidden_states_199)
        output_301 = None
        linear_175 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_75 = linear_175.view((1, 20, -1, 256))
        linear_175 = None
        query_states_50 = view_75.transpose(1, 2)
        view_75 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_76 = linear_176.view((1, 20, -1, 256))
        linear_176 = None
        key_states_50 = view_76.transpose(1, 2)
        view_76 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_200,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_200 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_77 = linear_177.view((1, 20, -1, 256))
        linear_177 = None
        value_states_25 = view_77.transpose(1, 2)
        view_77 = None
        float_311 = query_states_50.float()
        pow_152 = float_311.pow(2)
        mean_151 = pow_152.mean(-1, keepdim=True)
        pow_152 = None
        add_402 = mean_151 + 1e-06
        mean_151 = None
        rsqrt_151 = torch.rsqrt(add_402)
        add_402 = None
        output_302 = float_311 * rsqrt_151
        float_311 = rsqrt_151 = None
        float_312 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = (
            None
        )
        add_403 = 1.0 + float_312
        float_312 = None
        output_303 = output_302 * add_403
        output_302 = add_403 = None
        query_states_51 = output_303.type_as(query_states_50)
        output_303 = query_states_50 = None
        float_313 = key_states_50.float()
        pow_153 = float_313.pow(2)
        mean_152 = pow_153.mean(-1, keepdim=True)
        pow_153 = None
        add_404 = mean_152 + 1e-06
        mean_152 = None
        rsqrt_152 = torch.rsqrt(add_404)
        add_404 = None
        output_304 = float_313 * rsqrt_152
        float_313 = rsqrt_152 = None
        float_314 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = (
            None
        )
        add_405 = 1.0 + float_314
        float_314 = None
        output_305 = output_304 * add_405
        output_304 = add_405 = None
        key_states_51 = output_305.type_as(key_states_50)
        output_305 = key_states_50 = None
        cos_31 = cos_4.unsqueeze(1)
        cos_4 = None
        sin_31 = sin_4.unsqueeze(1)
        sin_4 = None
        mul_436 = query_states_51 * cos_31
        x1_50 = query_states_51[(Ellipsis, slice(None, 128, None))]
        x2_50 = query_states_51[(Ellipsis, slice(128, None, None))]
        query_states_51 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_52 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_437 = cat_52 * sin_31
        cat_52 = None
        q_embed_25 = mul_436 + mul_437
        mul_436 = mul_437 = None
        mul_438 = key_states_51 * cos_31
        cos_31 = None
        x1_51 = key_states_51[(Ellipsis, slice(None, 128, None))]
        x2_51 = key_states_51[(Ellipsis, slice(128, None, None))]
        key_states_51 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_53 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_439 = cat_53 * sin_31
        cat_53 = sin_31 = None
        k_embed_25 = mul_438 + mul_439
        mul_438 = mul_439 = None
        getitem_183 = k_embed_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        k_embed_25 = None
        hidden_states_201 = getitem_183.expand(1, 1, 4, 20, 256)
        getitem_183 = None
        key_50 = hidden_states_201.reshape(1, 4, 20, 256)
        hidden_states_201 = None
        getitem_184 = value_states_25[
            (
                slice(None, None, None),
                slice(None, None, None),
                None,
                slice(None, None, None),
                slice(None, None, None),
            )
        ]
        value_states_25 = None
        hidden_states_202 = getitem_184.expand(1, 1, 4, 20, 256)
        getitem_184 = None
        value_50 = hidden_states_202.reshape(1, 4, 20, 256)
        hidden_states_202 = None
        attention_mask_27 = causal_mask_1[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 20, None),
            )
        ]
        causal_mask_1 = None
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
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=0.0625,
            is_causal=False,
        )
        query_25 = key_51 = value_51 = attention_mask_27 = None
        transpose_105 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_105.contiguous()
        transpose_105 = None
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
        float_315 = attn_output_103.float()
        pow_154 = float_315.pow(2)
        mean_153 = pow_154.mean(-1, keepdim=True)
        pow_154 = None
        add_408 = mean_153 + 1e-06
        mean_153 = None
        rsqrt_153 = torch.rsqrt(add_408)
        add_408 = None
        output_306 = float_315 * rsqrt_153
        float_315 = rsqrt_153 = None
        float_316 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            None
        )
        add_409 = 1.0 + float_316
        float_316 = None
        output_307 = output_306 * add_409
        output_306 = add_409 = None
        hidden_states_203 = output_307.type_as(attn_output_103)
        output_307 = attn_output_103 = None
        hidden_states_204 = hidden_states_199 + hidden_states_203
        hidden_states_199 = hidden_states_203 = None
        float_317 = hidden_states_204.float()
        pow_155 = float_317.pow(2)
        mean_154 = pow_155.mean(-1, keepdim=True)
        pow_155 = None
        add_411 = mean_154 + 1e-06
        mean_154 = None
        rsqrt_154 = torch.rsqrt(add_411)
        add_411 = None
        output_308 = float_317 * rsqrt_154
        float_317 = rsqrt_154 = None
        float_318 = (
            l_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_412 = 1.0 + float_318
        float_318 = None
        output_309 = output_308 * add_412
        output_308 = add_412 = None
        hidden_states_205 = output_309.type_as(hidden_states_204)
        output_309 = None
        linear_179 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        gelu_25 = torch._C._nn.gelu(linear_179, approximate="tanh")
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_205 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_444 = gelu_25 * linear_180
        gelu_25 = linear_180 = None
        down_proj_25 = torch._C._nn.linear(
            mul_444,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_444 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        float_319 = down_proj_25.float()
        pow_156 = float_319.pow(2)
        mean_155 = pow_156.mean(-1, keepdim=True)
        pow_156 = None
        add_413 = mean_155 + 1e-06
        mean_155 = None
        rsqrt_155 = torch.rsqrt(add_413)
        add_413 = None
        output_310 = float_319 * rsqrt_155
        float_319 = rsqrt_155 = None
        float_320 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_.float()
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_ = (
            None
        )
        add_414 = 1.0 + float_320
        float_320 = None
        output_311 = output_310 * add_414
        output_310 = add_414 = None
        hidden_states_206 = output_311.type_as(down_proj_25)
        output_311 = down_proj_25 = None
        hidden_states_207 = hidden_states_204 + hidden_states_206
        hidden_states_204 = hidden_states_206 = None
        float_321 = hidden_states_207.float()
        pow_157 = float_321.pow(2)
        mean_156 = pow_157.mean(-1, keepdim=True)
        pow_157 = None
        add_416 = mean_156 + 1e-06
        mean_156 = None
        rsqrt_156 = torch.rsqrt(add_416)
        add_416 = None
        output_312 = float_321 * rsqrt_156
        float_321 = rsqrt_156 = None
        float_322 = l_self_modules_model_modules_norm_parameters_weight_.float()
        l_self_modules_model_modules_norm_parameters_weight_ = None
        add_417 = 1.0 + float_322
        float_322 = None
        output_313 = output_312 * add_417
        output_312 = add_417 = None
        hidden_states_208 = output_313.type_as(hidden_states_207)
        output_313 = hidden_states_207 = None
        getitem_186 = hidden_states_208[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_208 = None
        logits = torch._C._nn.linear(
            getitem_186,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
        )
        getitem_186 = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        return (logits,)
