import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_kwargs_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_rotary_emb_buffers_inv_freq_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_scaling: torch.Tensor,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_variance_epsilon: torch.Tensor,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_post_feedforward_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_variance_epsilon: torch.Tensor,
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
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_8_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_9_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_10_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_11_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_12_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_13_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_14_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_15_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_16_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_17_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_18_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_19_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_20_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_21_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_22_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_23_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_24_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_25_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_26_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_scaling = (
            L_self_modules_model_modules_layers_modules_27_modules_self_attn_scaling
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_variance_epsilon = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_variance_epsilon
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_norm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_post_feedforward_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_post_feedforward_layernorm_parameters_weight_
        l_self_modules_model_modules_norm_variance_epsilon = (
            L_self_modules_model_modules_norm_variance_epsilon
        )
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        inputs_embeds = torch.nn.functional.embedding(
            l_kwargs_input_ids_,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            0,
            None,
            2.0,
            False,
            False,
        )
        l_kwargs_input_ids_ = None
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
        linear = torch._C._nn.linear(
            inputs_embeds,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view = linear.view((1, 19, -1, 64))
        linear = None
        query_states = view.transpose(1, 2)
        view = None
        linear_1 = torch._C._nn.linear(
            inputs_embeds,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_1 = linear_1.view((1, 19, -1, 64))
        linear_1 = None
        key_states = view_1.transpose(1, 2)
        view_1 = None
        linear_2 = torch._C._nn.linear(
            inputs_embeds,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_2 = linear_2.view((1, 19, -1, 64))
        linear_2 = None
        value_states = view_2.transpose(1, 2)
        view_2 = None
        hidden_states = query_states.to(torch.float32)
        query_states = None
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
        query_states_1 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_
            * to_5
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_5
        ) = None
        hidden_states_2 = key_states.to(torch.float32)
        key_states = None
        pow_2 = hidden_states_2.pow(2)
        variance_1 = pow_2.mean(-1, keepdim=True)
        pow_2 = None
        add_1 = variance_1 + 1e-05
        variance_1 = None
        rsqrt_1 = torch.rsqrt(add_1)
        add_1 = None
        hidden_states_3 = hidden_states_2 * rsqrt_1
        hidden_states_2 = rsqrt_1 = None
        to_7 = hidden_states_3.to(torch.float16)
        hidden_states_3 = None
        key_states_1 = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_
            * to_7
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_7
        ) = None
        cos_3 = cos_2.unsqueeze(1)
        sin_3 = sin_2.unsqueeze(1)
        mul_6 = query_states_1 * cos_3
        x1 = query_states_1[(Ellipsis, slice(None, 32, None))]
        x2 = query_states_1[(Ellipsis, slice(32, None, None))]
        query_states_1 = None
        neg = -x2
        x2 = None
        cat_1 = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_7 = cat_1 * sin_3
        cat_1 = None
        q_embed = mul_6 + mul_7
        mul_6 = mul_7 = None
        mul_8 = key_states_1 * cos_3
        cos_3 = None
        x1_1 = key_states_1[(Ellipsis, slice(None, 32, None))]
        x2_1 = key_states_1[(Ellipsis, slice(32, None, None))]
        key_states_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_2 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_9 = cat_2 * sin_3
        cat_2 = sin_3 = None
        k_embed = mul_8 + mul_9
        mul_8 = mul_9 = None
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
        hidden_states_4 = getitem_6.expand(1, 8, 4, 19, 64)
        getitem_6 = None
        key = hidden_states_4.reshape(1, 32, 19, 64)
        hidden_states_4 = None
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
        hidden_states_5 = getitem_7.expand(1, 8, 4, 19, 64)
        getitem_7 = None
        value = hidden_states_5.reshape(1, 32, 19, 64)
        hidden_states_5 = None
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
        item = (
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_scaling = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
            query,
            key_1,
            value_1,
            attn_mask=attention_mask_1,
            dropout_p=0.0,
            scale=item,
            is_causal=False,
        )
        query = key_1 = value_1 = attention_mask_1 = item = None
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
        hidden_states_6 = attn_output_3.to(torch.float32)
        attn_output_3 = None
        pow_3 = hidden_states_6.pow(2)
        variance_2 = pow_3.mean(-1, keepdim=True)
        pow_3 = None
        item_1 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_4 = variance_2 + item_1
        variance_2 = item_1 = None
        rsqrt_2 = torch.rsqrt(add_4)
        add_4 = None
        hidden_states_7 = hidden_states_6 * rsqrt_2
        hidden_states_6 = rsqrt_2 = None
        to_9 = hidden_states_7.to(torch.float16)
        hidden_states_7 = None
        hidden_states_8 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
            * to_9
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = (
            to_9
        ) = None
        hidden_states_9 = inputs_embeds + hidden_states_8
        inputs_embeds = hidden_states_8 = None
        linear_4 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_9,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_12 = silu * linear_5
        silu = linear_5 = None
        down_proj = torch._C._nn.linear(
            mul_12,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_12 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_10 = down_proj.to(torch.float32)
        down_proj = None
        pow_4 = hidden_states_10.pow(2)
        variance_3 = pow_4.mean(-1, keepdim=True)
        pow_4 = None
        add_6 = variance_3 + 1e-05
        variance_3 = None
        rsqrt_3 = torch.rsqrt(add_6)
        add_6 = None
        hidden_states_11 = hidden_states_10 * rsqrt_3
        hidden_states_10 = rsqrt_3 = None
        to_11 = hidden_states_11.to(torch.float16)
        hidden_states_11 = None
        hidden_states_12 = (
            l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_
            * to_11
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_11
        ) = None
        hidden_states_13 = hidden_states_9 + hidden_states_12
        hidden_states_9 = hidden_states_12 = None
        linear_7 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_3 = linear_7.view((1, 19, -1, 64))
        linear_7 = None
        query_states_2 = view_3.transpose(1, 2)
        view_3 = None
        linear_8 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_4 = linear_8.view((1, 19, -1, 64))
        linear_8 = None
        key_states_2 = view_4.transpose(1, 2)
        view_4 = None
        linear_9 = torch._C._nn.linear(
            hidden_states_13,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_5 = linear_9.view((1, 19, -1, 64))
        linear_9 = None
        value_states_1 = view_5.transpose(1, 2)
        view_5 = None
        hidden_states_14 = query_states_2.to(torch.float32)
        query_states_2 = None
        pow_5 = hidden_states_14.pow(2)
        variance_4 = pow_5.mean(-1, keepdim=True)
        pow_5 = None
        add_8 = variance_4 + 1e-05
        variance_4 = None
        rsqrt_4 = torch.rsqrt(add_8)
        add_8 = None
        hidden_states_15 = hidden_states_14 * rsqrt_4
        hidden_states_14 = rsqrt_4 = None
        to_13 = hidden_states_15.to(torch.float16)
        hidden_states_15 = None
        query_states_3 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_
            * to_13
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_13
        ) = None
        hidden_states_16 = key_states_2.to(torch.float32)
        key_states_2 = None
        pow_6 = hidden_states_16.pow(2)
        variance_5 = pow_6.mean(-1, keepdim=True)
        pow_6 = None
        add_9 = variance_5 + 1e-05
        variance_5 = None
        rsqrt_5 = torch.rsqrt(add_9)
        add_9 = None
        hidden_states_17 = hidden_states_16 * rsqrt_5
        hidden_states_16 = rsqrt_5 = None
        to_15 = hidden_states_17.to(torch.float16)
        hidden_states_17 = None
        key_states_3 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_
            * to_15
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_15
        ) = None
        cos_4 = cos_2.unsqueeze(1)
        sin_4 = sin_2.unsqueeze(1)
        mul_19 = query_states_3 * cos_4
        x1_2 = query_states_3[(Ellipsis, slice(None, 32, None))]
        x2_2 = query_states_3[(Ellipsis, slice(32, None, None))]
        query_states_3 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_3 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_20 = cat_3 * sin_4
        cat_3 = None
        q_embed_1 = mul_19 + mul_20
        mul_19 = mul_20 = None
        mul_21 = key_states_3 * cos_4
        cos_4 = None
        x1_3 = key_states_3[(Ellipsis, slice(None, 32, None))]
        x2_3 = key_states_3[(Ellipsis, slice(32, None, None))]
        key_states_3 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_4 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_22 = cat_4 * sin_4
        cat_4 = sin_4 = None
        k_embed_1 = mul_21 + mul_22
        mul_21 = mul_22 = None
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
        hidden_states_18 = getitem_13.expand(1, 8, 4, 19, 64)
        getitem_13 = None
        key_2 = hidden_states_18.reshape(1, 32, 19, 64)
        hidden_states_18 = None
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
        hidden_states_19 = getitem_14.expand(1, 8, 4, 19, 64)
        getitem_14 = None
        value_2 = hidden_states_19.reshape(1, 32, 19, 64)
        hidden_states_19 = None
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
        item_2 = (
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_scaling = None
        attn_output_4 = torch._C._nn.scaled_dot_product_attention(
            query_1,
            key_3,
            value_3,
            attn_mask=attention_mask_2,
            dropout_p=0.0,
            scale=item_2,
            is_causal=False,
        )
        query_1 = key_3 = value_3 = attention_mask_2 = item_2 = None
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
        hidden_states_20 = attn_output_7.to(torch.float32)
        attn_output_7 = None
        pow_7 = hidden_states_20.pow(2)
        variance_6 = pow_7.mean(-1, keepdim=True)
        pow_7 = None
        item_3 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_12 = variance_6 + item_3
        variance_6 = item_3 = None
        rsqrt_6 = torch.rsqrt(add_12)
        add_12 = None
        hidden_states_21 = hidden_states_20 * rsqrt_6
        hidden_states_20 = rsqrt_6 = None
        to_17 = hidden_states_21.to(torch.float16)
        hidden_states_21 = None
        hidden_states_22 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
            * to_17
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = (
            to_17
        ) = None
        hidden_states_23 = hidden_states_13 + hidden_states_22
        hidden_states_13 = hidden_states_22 = None
        linear_11 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_23,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_25 = silu_1 * linear_12
        silu_1 = linear_12 = None
        down_proj_1 = torch._C._nn.linear(
            mul_25,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_25 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_24 = down_proj_1.to(torch.float32)
        down_proj_1 = None
        pow_8 = hidden_states_24.pow(2)
        variance_7 = pow_8.mean(-1, keepdim=True)
        pow_8 = None
        add_14 = variance_7 + 1e-05
        variance_7 = None
        rsqrt_7 = torch.rsqrt(add_14)
        add_14 = None
        hidden_states_25 = hidden_states_24 * rsqrt_7
        hidden_states_24 = rsqrt_7 = None
        to_19 = hidden_states_25.to(torch.float16)
        hidden_states_25 = None
        hidden_states_26 = (
            l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_
            * to_19
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_19
        ) = None
        hidden_states_27 = hidden_states_23 + hidden_states_26
        hidden_states_23 = hidden_states_26 = None
        linear_14 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_6 = linear_14.view((1, 19, -1, 64))
        linear_14 = None
        query_states_4 = view_6.transpose(1, 2)
        view_6 = None
        linear_15 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_7 = linear_15.view((1, 19, -1, 64))
        linear_15 = None
        key_states_4 = view_7.transpose(1, 2)
        view_7 = None
        linear_16 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_8 = linear_16.view((1, 19, -1, 64))
        linear_16 = None
        value_states_2 = view_8.transpose(1, 2)
        view_8 = None
        hidden_states_28 = query_states_4.to(torch.float32)
        query_states_4 = None
        pow_9 = hidden_states_28.pow(2)
        variance_8 = pow_9.mean(-1, keepdim=True)
        pow_9 = None
        add_16 = variance_8 + 1e-05
        variance_8 = None
        rsqrt_8 = torch.rsqrt(add_16)
        add_16 = None
        hidden_states_29 = hidden_states_28 * rsqrt_8
        hidden_states_28 = rsqrt_8 = None
        to_21 = hidden_states_29.to(torch.float16)
        hidden_states_29 = None
        query_states_5 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_
            * to_21
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_21
        ) = None
        hidden_states_30 = key_states_4.to(torch.float32)
        key_states_4 = None
        pow_10 = hidden_states_30.pow(2)
        variance_9 = pow_10.mean(-1, keepdim=True)
        pow_10 = None
        add_17 = variance_9 + 1e-05
        variance_9 = None
        rsqrt_9 = torch.rsqrt(add_17)
        add_17 = None
        hidden_states_31 = hidden_states_30 * rsqrt_9
        hidden_states_30 = rsqrt_9 = None
        to_23 = hidden_states_31.to(torch.float16)
        hidden_states_31 = None
        key_states_5 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_
            * to_23
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_23
        ) = None
        cos_5 = cos_2.unsqueeze(1)
        sin_5 = sin_2.unsqueeze(1)
        mul_32 = query_states_5 * cos_5
        x1_4 = query_states_5[(Ellipsis, slice(None, 32, None))]
        x2_4 = query_states_5[(Ellipsis, slice(32, None, None))]
        query_states_5 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_5 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_33 = cat_5 * sin_5
        cat_5 = None
        q_embed_2 = mul_32 + mul_33
        mul_32 = mul_33 = None
        mul_34 = key_states_5 * cos_5
        cos_5 = None
        x1_5 = key_states_5[(Ellipsis, slice(None, 32, None))]
        x2_5 = key_states_5[(Ellipsis, slice(32, None, None))]
        key_states_5 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_6 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_35 = cat_6 * sin_5
        cat_6 = sin_5 = None
        k_embed_2 = mul_34 + mul_35
        mul_34 = mul_35 = None
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
        hidden_states_32 = getitem_20.expand(1, 8, 4, 19, 64)
        getitem_20 = None
        key_4 = hidden_states_32.reshape(1, 32, 19, 64)
        hidden_states_32 = None
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
        hidden_states_33 = getitem_21.expand(1, 8, 4, 19, 64)
        getitem_21 = None
        value_4 = hidden_states_33.reshape(1, 32, 19, 64)
        hidden_states_33 = None
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
        item_4 = (
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_scaling = None
        attn_output_8 = torch._C._nn.scaled_dot_product_attention(
            query_2,
            key_5,
            value_5,
            attn_mask=attention_mask_3,
            dropout_p=0.0,
            scale=item_4,
            is_causal=False,
        )
        query_2 = key_5 = value_5 = attention_mask_3 = item_4 = None
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
        hidden_states_34 = attn_output_11.to(torch.float32)
        attn_output_11 = None
        pow_11 = hidden_states_34.pow(2)
        variance_10 = pow_11.mean(-1, keepdim=True)
        pow_11 = None
        item_5 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_20 = variance_10 + item_5
        variance_10 = item_5 = None
        rsqrt_10 = torch.rsqrt(add_20)
        add_20 = None
        hidden_states_35 = hidden_states_34 * rsqrt_10
        hidden_states_34 = rsqrt_10 = None
        to_25 = hidden_states_35.to(torch.float16)
        hidden_states_35 = None
        hidden_states_36 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
            * to_25
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = (
            to_25
        ) = None
        hidden_states_37 = hidden_states_27 + hidden_states_36
        hidden_states_27 = hidden_states_36 = None
        linear_18 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_38 = silu_2 * linear_19
        silu_2 = linear_19 = None
        down_proj_2 = torch._C._nn.linear(
            mul_38,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_38 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_38 = down_proj_2.to(torch.float32)
        down_proj_2 = None
        pow_12 = hidden_states_38.pow(2)
        variance_11 = pow_12.mean(-1, keepdim=True)
        pow_12 = None
        add_22 = variance_11 + 1e-05
        variance_11 = None
        rsqrt_11 = torch.rsqrt(add_22)
        add_22 = None
        hidden_states_39 = hidden_states_38 * rsqrt_11
        hidden_states_38 = rsqrt_11 = None
        to_27 = hidden_states_39.to(torch.float16)
        hidden_states_39 = None
        hidden_states_40 = (
            l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_
            * to_27
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_27
        ) = None
        hidden_states_41 = hidden_states_37 + hidden_states_40
        hidden_states_37 = hidden_states_40 = None
        linear_21 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_9 = linear_21.view((1, 19, -1, 64))
        linear_21 = None
        query_states_6 = view_9.transpose(1, 2)
        view_9 = None
        linear_22 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_10 = linear_22.view((1, 19, -1, 64))
        linear_22 = None
        key_states_6 = view_10.transpose(1, 2)
        view_10 = None
        linear_23 = torch._C._nn.linear(
            hidden_states_41,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_11 = linear_23.view((1, 19, -1, 64))
        linear_23 = None
        value_states_3 = view_11.transpose(1, 2)
        view_11 = None
        hidden_states_42 = query_states_6.to(torch.float32)
        query_states_6 = None
        pow_13 = hidden_states_42.pow(2)
        variance_12 = pow_13.mean(-1, keepdim=True)
        pow_13 = None
        add_24 = variance_12 + 1e-05
        variance_12 = None
        rsqrt_12 = torch.rsqrt(add_24)
        add_24 = None
        hidden_states_43 = hidden_states_42 * rsqrt_12
        hidden_states_42 = rsqrt_12 = None
        to_29 = hidden_states_43.to(torch.float16)
        hidden_states_43 = None
        query_states_7 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_
            * to_29
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_29
        ) = None
        hidden_states_44 = key_states_6.to(torch.float32)
        key_states_6 = None
        pow_14 = hidden_states_44.pow(2)
        variance_13 = pow_14.mean(-1, keepdim=True)
        pow_14 = None
        add_25 = variance_13 + 1e-05
        variance_13 = None
        rsqrt_13 = torch.rsqrt(add_25)
        add_25 = None
        hidden_states_45 = hidden_states_44 * rsqrt_13
        hidden_states_44 = rsqrt_13 = None
        to_31 = hidden_states_45.to(torch.float16)
        hidden_states_45 = None
        key_states_7 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_
            * to_31
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_31
        ) = None
        cos_6 = cos_2.unsqueeze(1)
        sin_6 = sin_2.unsqueeze(1)
        mul_45 = query_states_7 * cos_6
        x1_6 = query_states_7[(Ellipsis, slice(None, 32, None))]
        x2_6 = query_states_7[(Ellipsis, slice(32, None, None))]
        query_states_7 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_7 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_46 = cat_7 * sin_6
        cat_7 = None
        q_embed_3 = mul_45 + mul_46
        mul_45 = mul_46 = None
        mul_47 = key_states_7 * cos_6
        cos_6 = None
        x1_7 = key_states_7[(Ellipsis, slice(None, 32, None))]
        x2_7 = key_states_7[(Ellipsis, slice(32, None, None))]
        key_states_7 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_8 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_48 = cat_8 * sin_6
        cat_8 = sin_6 = None
        k_embed_3 = mul_47 + mul_48
        mul_47 = mul_48 = None
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
        hidden_states_46 = getitem_27.expand(1, 8, 4, 19, 64)
        getitem_27 = None
        key_6 = hidden_states_46.reshape(1, 32, 19, 64)
        hidden_states_46 = None
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
        hidden_states_47 = getitem_28.expand(1, 8, 4, 19, 64)
        getitem_28 = None
        value_6 = hidden_states_47.reshape(1, 32, 19, 64)
        hidden_states_47 = None
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
        item_6 = (
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_scaling = None
        attn_output_12 = torch._C._nn.scaled_dot_product_attention(
            query_3,
            key_7,
            value_7,
            attn_mask=attention_mask_4,
            dropout_p=0.0,
            scale=item_6,
            is_causal=False,
        )
        query_3 = key_7 = value_7 = attention_mask_4 = item_6 = None
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
        hidden_states_48 = attn_output_15.to(torch.float32)
        attn_output_15 = None
        pow_15 = hidden_states_48.pow(2)
        variance_14 = pow_15.mean(-1, keepdim=True)
        pow_15 = None
        item_7 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_28 = variance_14 + item_7
        variance_14 = item_7 = None
        rsqrt_14 = torch.rsqrt(add_28)
        add_28 = None
        hidden_states_49 = hidden_states_48 * rsqrt_14
        hidden_states_48 = rsqrt_14 = None
        to_33 = hidden_states_49.to(torch.float16)
        hidden_states_49 = None
        hidden_states_50 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
            * to_33
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = (
            to_33
        ) = None
        hidden_states_51 = hidden_states_41 + hidden_states_50
        hidden_states_41 = hidden_states_50 = None
        linear_25 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_51,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_51 = silu_3 * linear_26
        silu_3 = linear_26 = None
        down_proj_3 = torch._C._nn.linear(
            mul_51,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_51 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_52 = down_proj_3.to(torch.float32)
        down_proj_3 = None
        pow_16 = hidden_states_52.pow(2)
        variance_15 = pow_16.mean(-1, keepdim=True)
        pow_16 = None
        add_30 = variance_15 + 1e-05
        variance_15 = None
        rsqrt_15 = torch.rsqrt(add_30)
        add_30 = None
        hidden_states_53 = hidden_states_52 * rsqrt_15
        hidden_states_52 = rsqrt_15 = None
        to_35 = hidden_states_53.to(torch.float16)
        hidden_states_53 = None
        hidden_states_54 = (
            l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_
            * to_35
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_35
        ) = None
        hidden_states_55 = hidden_states_51 + hidden_states_54
        hidden_states_51 = hidden_states_54 = None
        linear_28 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_12 = linear_28.view((1, 19, -1, 64))
        linear_28 = None
        query_states_8 = view_12.transpose(1, 2)
        view_12 = None
        linear_29 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_13 = linear_29.view((1, 19, -1, 64))
        linear_29 = None
        key_states_8 = view_13.transpose(1, 2)
        view_13 = None
        linear_30 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_14 = linear_30.view((1, 19, -1, 64))
        linear_30 = None
        value_states_4 = view_14.transpose(1, 2)
        view_14 = None
        hidden_states_56 = query_states_8.to(torch.float32)
        query_states_8 = None
        pow_17 = hidden_states_56.pow(2)
        variance_16 = pow_17.mean(-1, keepdim=True)
        pow_17 = None
        add_32 = variance_16 + 1e-05
        variance_16 = None
        rsqrt_16 = torch.rsqrt(add_32)
        add_32 = None
        hidden_states_57 = hidden_states_56 * rsqrt_16
        hidden_states_56 = rsqrt_16 = None
        to_37 = hidden_states_57.to(torch.float16)
        hidden_states_57 = None
        query_states_9 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_
            * to_37
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_37
        ) = None
        hidden_states_58 = key_states_8.to(torch.float32)
        key_states_8 = None
        pow_18 = hidden_states_58.pow(2)
        variance_17 = pow_18.mean(-1, keepdim=True)
        pow_18 = None
        add_33 = variance_17 + 1e-05
        variance_17 = None
        rsqrt_17 = torch.rsqrt(add_33)
        add_33 = None
        hidden_states_59 = hidden_states_58 * rsqrt_17
        hidden_states_58 = rsqrt_17 = None
        to_39 = hidden_states_59.to(torch.float16)
        hidden_states_59 = None
        key_states_9 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_
            * to_39
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_39
        ) = None
        cos_7 = cos_2.unsqueeze(1)
        sin_7 = sin_2.unsqueeze(1)
        mul_58 = query_states_9 * cos_7
        x1_8 = query_states_9[(Ellipsis, slice(None, 32, None))]
        x2_8 = query_states_9[(Ellipsis, slice(32, None, None))]
        query_states_9 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_9 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_59 = cat_9 * sin_7
        cat_9 = None
        q_embed_4 = mul_58 + mul_59
        mul_58 = mul_59 = None
        mul_60 = key_states_9 * cos_7
        cos_7 = None
        x1_9 = key_states_9[(Ellipsis, slice(None, 32, None))]
        x2_9 = key_states_9[(Ellipsis, slice(32, None, None))]
        key_states_9 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_10 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_61 = cat_10 * sin_7
        cat_10 = sin_7 = None
        k_embed_4 = mul_60 + mul_61
        mul_60 = mul_61 = None
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
        hidden_states_60 = getitem_34.expand(1, 8, 4, 19, 64)
        getitem_34 = None
        key_8 = hidden_states_60.reshape(1, 32, 19, 64)
        hidden_states_60 = None
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
        hidden_states_61 = getitem_35.expand(1, 8, 4, 19, 64)
        getitem_35 = None
        value_8 = hidden_states_61.reshape(1, 32, 19, 64)
        hidden_states_61 = None
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
        item_8 = (
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_scaling = None
        attn_output_16 = torch._C._nn.scaled_dot_product_attention(
            query_4,
            key_9,
            value_9,
            attn_mask=attention_mask_5,
            dropout_p=0.0,
            scale=item_8,
            is_causal=False,
        )
        query_4 = key_9 = value_9 = attention_mask_5 = item_8 = None
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
        hidden_states_62 = attn_output_19.to(torch.float32)
        attn_output_19 = None
        pow_19 = hidden_states_62.pow(2)
        variance_18 = pow_19.mean(-1, keepdim=True)
        pow_19 = None
        item_9 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_36 = variance_18 + item_9
        variance_18 = item_9 = None
        rsqrt_18 = torch.rsqrt(add_36)
        add_36 = None
        hidden_states_63 = hidden_states_62 * rsqrt_18
        hidden_states_62 = rsqrt_18 = None
        to_41 = hidden_states_63.to(torch.float16)
        hidden_states_63 = None
        hidden_states_64 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
            * to_41
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = (
            to_41
        ) = None
        hidden_states_65 = hidden_states_55 + hidden_states_64
        hidden_states_55 = hidden_states_64 = None
        linear_32 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_64 = silu_4 * linear_33
        silu_4 = linear_33 = None
        down_proj_4 = torch._C._nn.linear(
            mul_64,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_64 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_66 = down_proj_4.to(torch.float32)
        down_proj_4 = None
        pow_20 = hidden_states_66.pow(2)
        variance_19 = pow_20.mean(-1, keepdim=True)
        pow_20 = None
        add_38 = variance_19 + 1e-05
        variance_19 = None
        rsqrt_19 = torch.rsqrt(add_38)
        add_38 = None
        hidden_states_67 = hidden_states_66 * rsqrt_19
        hidden_states_66 = rsqrt_19 = None
        to_43 = hidden_states_67.to(torch.float16)
        hidden_states_67 = None
        hidden_states_68 = (
            l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_
            * to_43
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_43
        ) = None
        hidden_states_69 = hidden_states_65 + hidden_states_68
        hidden_states_65 = hidden_states_68 = None
        linear_35 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_15 = linear_35.view((1, 19, -1, 64))
        linear_35 = None
        query_states_10 = view_15.transpose(1, 2)
        view_15 = None
        linear_36 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_16 = linear_36.view((1, 19, -1, 64))
        linear_36 = None
        key_states_10 = view_16.transpose(1, 2)
        view_16 = None
        linear_37 = torch._C._nn.linear(
            hidden_states_69,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_17 = linear_37.view((1, 19, -1, 64))
        linear_37 = None
        value_states_5 = view_17.transpose(1, 2)
        view_17 = None
        hidden_states_70 = query_states_10.to(torch.float32)
        query_states_10 = None
        pow_21 = hidden_states_70.pow(2)
        variance_20 = pow_21.mean(-1, keepdim=True)
        pow_21 = None
        add_40 = variance_20 + 1e-05
        variance_20 = None
        rsqrt_20 = torch.rsqrt(add_40)
        add_40 = None
        hidden_states_71 = hidden_states_70 * rsqrt_20
        hidden_states_70 = rsqrt_20 = None
        to_45 = hidden_states_71.to(torch.float16)
        hidden_states_71 = None
        query_states_11 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_
            * to_45
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_45
        ) = None
        hidden_states_72 = key_states_10.to(torch.float32)
        key_states_10 = None
        pow_22 = hidden_states_72.pow(2)
        variance_21 = pow_22.mean(-1, keepdim=True)
        pow_22 = None
        add_41 = variance_21 + 1e-05
        variance_21 = None
        rsqrt_21 = torch.rsqrt(add_41)
        add_41 = None
        hidden_states_73 = hidden_states_72 * rsqrt_21
        hidden_states_72 = rsqrt_21 = None
        to_47 = hidden_states_73.to(torch.float16)
        hidden_states_73 = None
        key_states_11 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_
            * to_47
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_47
        ) = None
        cos_8 = cos_2.unsqueeze(1)
        sin_8 = sin_2.unsqueeze(1)
        mul_71 = query_states_11 * cos_8
        x1_10 = query_states_11[(Ellipsis, slice(None, 32, None))]
        x2_10 = query_states_11[(Ellipsis, slice(32, None, None))]
        query_states_11 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_11 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_72 = cat_11 * sin_8
        cat_11 = None
        q_embed_5 = mul_71 + mul_72
        mul_71 = mul_72 = None
        mul_73 = key_states_11 * cos_8
        cos_8 = None
        x1_11 = key_states_11[(Ellipsis, slice(None, 32, None))]
        x2_11 = key_states_11[(Ellipsis, slice(32, None, None))]
        key_states_11 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_12 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_74 = cat_12 * sin_8
        cat_12 = sin_8 = None
        k_embed_5 = mul_73 + mul_74
        mul_73 = mul_74 = None
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
        hidden_states_74 = getitem_41.expand(1, 8, 4, 19, 64)
        getitem_41 = None
        key_10 = hidden_states_74.reshape(1, 32, 19, 64)
        hidden_states_74 = None
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
        hidden_states_75 = getitem_42.expand(1, 8, 4, 19, 64)
        getitem_42 = None
        value_10 = hidden_states_75.reshape(1, 32, 19, 64)
        hidden_states_75 = None
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
        item_10 = (
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_scaling = None
        attn_output_20 = torch._C._nn.scaled_dot_product_attention(
            query_5,
            key_11,
            value_11,
            attn_mask=attention_mask_6,
            dropout_p=0.0,
            scale=item_10,
            is_causal=False,
        )
        query_5 = key_11 = value_11 = attention_mask_6 = item_10 = None
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
        hidden_states_76 = attn_output_23.to(torch.float32)
        attn_output_23 = None
        pow_23 = hidden_states_76.pow(2)
        variance_22 = pow_23.mean(-1, keepdim=True)
        pow_23 = None
        item_11 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_44 = variance_22 + item_11
        variance_22 = item_11 = None
        rsqrt_22 = torch.rsqrt(add_44)
        add_44 = None
        hidden_states_77 = hidden_states_76 * rsqrt_22
        hidden_states_76 = rsqrt_22 = None
        to_49 = hidden_states_77.to(torch.float16)
        hidden_states_77 = None
        hidden_states_78 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
            * to_49
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = (
            to_49
        ) = None
        hidden_states_79 = hidden_states_69 + hidden_states_78
        hidden_states_69 = hidden_states_78 = None
        linear_39 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_79,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_77 = silu_5 * linear_40
        silu_5 = linear_40 = None
        down_proj_5 = torch._C._nn.linear(
            mul_77,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_77 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_80 = down_proj_5.to(torch.float32)
        down_proj_5 = None
        pow_24 = hidden_states_80.pow(2)
        variance_23 = pow_24.mean(-1, keepdim=True)
        pow_24 = None
        add_46 = variance_23 + 1e-05
        variance_23 = None
        rsqrt_23 = torch.rsqrt(add_46)
        add_46 = None
        hidden_states_81 = hidden_states_80 * rsqrt_23
        hidden_states_80 = rsqrt_23 = None
        to_51 = hidden_states_81.to(torch.float16)
        hidden_states_81 = None
        hidden_states_82 = (
            l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_
            * to_51
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_51
        ) = None
        hidden_states_83 = hidden_states_79 + hidden_states_82
        hidden_states_79 = hidden_states_82 = None
        linear_42 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_18 = linear_42.view((1, 19, -1, 64))
        linear_42 = None
        query_states_12 = view_18.transpose(1, 2)
        view_18 = None
        linear_43 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_19 = linear_43.view((1, 19, -1, 64))
        linear_43 = None
        key_states_12 = view_19.transpose(1, 2)
        view_19 = None
        linear_44 = torch._C._nn.linear(
            hidden_states_83,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_20 = linear_44.view((1, 19, -1, 64))
        linear_44 = None
        value_states_6 = view_20.transpose(1, 2)
        view_20 = None
        hidden_states_84 = query_states_12.to(torch.float32)
        query_states_12 = None
        pow_25 = hidden_states_84.pow(2)
        variance_24 = pow_25.mean(-1, keepdim=True)
        pow_25 = None
        add_48 = variance_24 + 1e-05
        variance_24 = None
        rsqrt_24 = torch.rsqrt(add_48)
        add_48 = None
        hidden_states_85 = hidden_states_84 * rsqrt_24
        hidden_states_84 = rsqrt_24 = None
        to_53 = hidden_states_85.to(torch.float16)
        hidden_states_85 = None
        query_states_13 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_
            * to_53
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_53
        ) = None
        hidden_states_86 = key_states_12.to(torch.float32)
        key_states_12 = None
        pow_26 = hidden_states_86.pow(2)
        variance_25 = pow_26.mean(-1, keepdim=True)
        pow_26 = None
        add_49 = variance_25 + 1e-05
        variance_25 = None
        rsqrt_25 = torch.rsqrt(add_49)
        add_49 = None
        hidden_states_87 = hidden_states_86 * rsqrt_25
        hidden_states_86 = rsqrt_25 = None
        to_55 = hidden_states_87.to(torch.float16)
        hidden_states_87 = None
        key_states_13 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_
            * to_55
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_55
        ) = None
        cos_9 = cos_2.unsqueeze(1)
        sin_9 = sin_2.unsqueeze(1)
        mul_84 = query_states_13 * cos_9
        x1_12 = query_states_13[(Ellipsis, slice(None, 32, None))]
        x2_12 = query_states_13[(Ellipsis, slice(32, None, None))]
        query_states_13 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_13 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_85 = cat_13 * sin_9
        cat_13 = None
        q_embed_6 = mul_84 + mul_85
        mul_84 = mul_85 = None
        mul_86 = key_states_13 * cos_9
        cos_9 = None
        x1_13 = key_states_13[(Ellipsis, slice(None, 32, None))]
        x2_13 = key_states_13[(Ellipsis, slice(32, None, None))]
        key_states_13 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_14 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_87 = cat_14 * sin_9
        cat_14 = sin_9 = None
        k_embed_6 = mul_86 + mul_87
        mul_86 = mul_87 = None
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
        hidden_states_88 = getitem_48.expand(1, 8, 4, 19, 64)
        getitem_48 = None
        key_12 = hidden_states_88.reshape(1, 32, 19, 64)
        hidden_states_88 = None
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
        hidden_states_89 = getitem_49.expand(1, 8, 4, 19, 64)
        getitem_49 = None
        value_12 = hidden_states_89.reshape(1, 32, 19, 64)
        hidden_states_89 = None
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
        item_12 = (
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_scaling = None
        attn_output_24 = torch._C._nn.scaled_dot_product_attention(
            query_6,
            key_13,
            value_13,
            attn_mask=attention_mask_7,
            dropout_p=0.0,
            scale=item_12,
            is_causal=False,
        )
        query_6 = key_13 = value_13 = attention_mask_7 = item_12 = None
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
        hidden_states_90 = attn_output_27.to(torch.float32)
        attn_output_27 = None
        pow_27 = hidden_states_90.pow(2)
        variance_26 = pow_27.mean(-1, keepdim=True)
        pow_27 = None
        item_13 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_52 = variance_26 + item_13
        variance_26 = item_13 = None
        rsqrt_26 = torch.rsqrt(add_52)
        add_52 = None
        hidden_states_91 = hidden_states_90 * rsqrt_26
        hidden_states_90 = rsqrt_26 = None
        to_57 = hidden_states_91.to(torch.float16)
        hidden_states_91 = None
        hidden_states_92 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
            * to_57
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = (
            to_57
        ) = None
        hidden_states_93 = hidden_states_83 + hidden_states_92
        hidden_states_83 = hidden_states_92 = None
        linear_46 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_93,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_90 = silu_6 * linear_47
        silu_6 = linear_47 = None
        down_proj_6 = torch._C._nn.linear(
            mul_90,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_90 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_94 = down_proj_6.to(torch.float32)
        down_proj_6 = None
        pow_28 = hidden_states_94.pow(2)
        variance_27 = pow_28.mean(-1, keepdim=True)
        pow_28 = None
        add_54 = variance_27 + 1e-05
        variance_27 = None
        rsqrt_27 = torch.rsqrt(add_54)
        add_54 = None
        hidden_states_95 = hidden_states_94 * rsqrt_27
        hidden_states_94 = rsqrt_27 = None
        to_59 = hidden_states_95.to(torch.float16)
        hidden_states_95 = None
        hidden_states_96 = (
            l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_
            * to_59
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_59
        ) = None
        hidden_states_97 = hidden_states_93 + hidden_states_96
        hidden_states_93 = hidden_states_96 = None
        linear_49 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_21 = linear_49.view((1, 19, -1, 64))
        linear_49 = None
        query_states_14 = view_21.transpose(1, 2)
        view_21 = None
        linear_50 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_22 = linear_50.view((1, 19, -1, 64))
        linear_50 = None
        key_states_14 = view_22.transpose(1, 2)
        view_22 = None
        linear_51 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_23 = linear_51.view((1, 19, -1, 64))
        linear_51 = None
        value_states_7 = view_23.transpose(1, 2)
        view_23 = None
        hidden_states_98 = query_states_14.to(torch.float32)
        query_states_14 = None
        pow_29 = hidden_states_98.pow(2)
        variance_28 = pow_29.mean(-1, keepdim=True)
        pow_29 = None
        add_56 = variance_28 + 1e-05
        variance_28 = None
        rsqrt_28 = torch.rsqrt(add_56)
        add_56 = None
        hidden_states_99 = hidden_states_98 * rsqrt_28
        hidden_states_98 = rsqrt_28 = None
        to_61 = hidden_states_99.to(torch.float16)
        hidden_states_99 = None
        query_states_15 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_
            * to_61
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_61
        ) = None
        hidden_states_100 = key_states_14.to(torch.float32)
        key_states_14 = None
        pow_30 = hidden_states_100.pow(2)
        variance_29 = pow_30.mean(-1, keepdim=True)
        pow_30 = None
        add_57 = variance_29 + 1e-05
        variance_29 = None
        rsqrt_29 = torch.rsqrt(add_57)
        add_57 = None
        hidden_states_101 = hidden_states_100 * rsqrt_29
        hidden_states_100 = rsqrt_29 = None
        to_63 = hidden_states_101.to(torch.float16)
        hidden_states_101 = None
        key_states_15 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_
            * to_63
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_63
        ) = None
        cos_10 = cos_2.unsqueeze(1)
        sin_10 = sin_2.unsqueeze(1)
        mul_97 = query_states_15 * cos_10
        x1_14 = query_states_15[(Ellipsis, slice(None, 32, None))]
        x2_14 = query_states_15[(Ellipsis, slice(32, None, None))]
        query_states_15 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_15 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_98 = cat_15 * sin_10
        cat_15 = None
        q_embed_7 = mul_97 + mul_98
        mul_97 = mul_98 = None
        mul_99 = key_states_15 * cos_10
        cos_10 = None
        x1_15 = key_states_15[(Ellipsis, slice(None, 32, None))]
        x2_15 = key_states_15[(Ellipsis, slice(32, None, None))]
        key_states_15 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_16 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_100 = cat_16 * sin_10
        cat_16 = sin_10 = None
        k_embed_7 = mul_99 + mul_100
        mul_99 = mul_100 = None
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
        hidden_states_102 = getitem_55.expand(1, 8, 4, 19, 64)
        getitem_55 = None
        key_14 = hidden_states_102.reshape(1, 32, 19, 64)
        hidden_states_102 = None
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
        hidden_states_103 = getitem_56.expand(1, 8, 4, 19, 64)
        getitem_56 = None
        value_14 = hidden_states_103.reshape(1, 32, 19, 64)
        hidden_states_103 = None
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
        item_14 = (
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_scaling = None
        attn_output_28 = torch._C._nn.scaled_dot_product_attention(
            query_7,
            key_15,
            value_15,
            attn_mask=attention_mask_8,
            dropout_p=0.0,
            scale=item_14,
            is_causal=False,
        )
        query_7 = key_15 = value_15 = attention_mask_8 = item_14 = None
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
        hidden_states_104 = attn_output_31.to(torch.float32)
        attn_output_31 = None
        pow_31 = hidden_states_104.pow(2)
        variance_30 = pow_31.mean(-1, keepdim=True)
        pow_31 = None
        item_15 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_60 = variance_30 + item_15
        variance_30 = item_15 = None
        rsqrt_30 = torch.rsqrt(add_60)
        add_60 = None
        hidden_states_105 = hidden_states_104 * rsqrt_30
        hidden_states_104 = rsqrt_30 = None
        to_65 = hidden_states_105.to(torch.float16)
        hidden_states_105 = None
        hidden_states_106 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
            * to_65
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = (
            to_65
        ) = None
        hidden_states_107 = hidden_states_97 + hidden_states_106
        hidden_states_97 = hidden_states_106 = None
        linear_53 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_103 = silu_7 * linear_54
        silu_7 = linear_54 = None
        down_proj_7 = torch._C._nn.linear(
            mul_103,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_103 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_108 = down_proj_7.to(torch.float32)
        down_proj_7 = None
        pow_32 = hidden_states_108.pow(2)
        variance_31 = pow_32.mean(-1, keepdim=True)
        pow_32 = None
        add_62 = variance_31 + 1e-05
        variance_31 = None
        rsqrt_31 = torch.rsqrt(add_62)
        add_62 = None
        hidden_states_109 = hidden_states_108 * rsqrt_31
        hidden_states_108 = rsqrt_31 = None
        to_67 = hidden_states_109.to(torch.float16)
        hidden_states_109 = None
        hidden_states_110 = (
            l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_
            * to_67
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_67
        ) = None
        hidden_states_111 = hidden_states_107 + hidden_states_110
        hidden_states_107 = hidden_states_110 = None
        linear_56 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_24 = linear_56.view((1, 19, -1, 64))
        linear_56 = None
        query_states_16 = view_24.transpose(1, 2)
        view_24 = None
        linear_57 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_25 = linear_57.view((1, 19, -1, 64))
        linear_57 = None
        key_states_16 = view_25.transpose(1, 2)
        view_25 = None
        linear_58 = torch._C._nn.linear(
            hidden_states_111,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_26 = linear_58.view((1, 19, -1, 64))
        linear_58 = None
        value_states_8 = view_26.transpose(1, 2)
        view_26 = None
        hidden_states_112 = query_states_16.to(torch.float32)
        query_states_16 = None
        pow_33 = hidden_states_112.pow(2)
        variance_32 = pow_33.mean(-1, keepdim=True)
        pow_33 = None
        add_64 = variance_32 + 1e-05
        variance_32 = None
        rsqrt_32 = torch.rsqrt(add_64)
        add_64 = None
        hidden_states_113 = hidden_states_112 * rsqrt_32
        hidden_states_112 = rsqrt_32 = None
        to_69 = hidden_states_113.to(torch.float16)
        hidden_states_113 = None
        query_states_17 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_
            * to_69
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_69
        ) = None
        hidden_states_114 = key_states_16.to(torch.float32)
        key_states_16 = None
        pow_34 = hidden_states_114.pow(2)
        variance_33 = pow_34.mean(-1, keepdim=True)
        pow_34 = None
        add_65 = variance_33 + 1e-05
        variance_33 = None
        rsqrt_33 = torch.rsqrt(add_65)
        add_65 = None
        hidden_states_115 = hidden_states_114 * rsqrt_33
        hidden_states_114 = rsqrt_33 = None
        to_71 = hidden_states_115.to(torch.float16)
        hidden_states_115 = None
        key_states_17 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_
            * to_71
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_71
        ) = None
        cos_11 = cos_2.unsqueeze(1)
        sin_11 = sin_2.unsqueeze(1)
        mul_110 = query_states_17 * cos_11
        x1_16 = query_states_17[(Ellipsis, slice(None, 32, None))]
        x2_16 = query_states_17[(Ellipsis, slice(32, None, None))]
        query_states_17 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_17 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_111 = cat_17 * sin_11
        cat_17 = None
        q_embed_8 = mul_110 + mul_111
        mul_110 = mul_111 = None
        mul_112 = key_states_17 * cos_11
        cos_11 = None
        x1_17 = key_states_17[(Ellipsis, slice(None, 32, None))]
        x2_17 = key_states_17[(Ellipsis, slice(32, None, None))]
        key_states_17 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_18 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_113 = cat_18 * sin_11
        cat_18 = sin_11 = None
        k_embed_8 = mul_112 + mul_113
        mul_112 = mul_113 = None
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
        hidden_states_116 = getitem_62.expand(1, 8, 4, 19, 64)
        getitem_62 = None
        key_16 = hidden_states_116.reshape(1, 32, 19, 64)
        hidden_states_116 = None
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
        hidden_states_117 = getitem_63.expand(1, 8, 4, 19, 64)
        getitem_63 = None
        value_16 = hidden_states_117.reshape(1, 32, 19, 64)
        hidden_states_117 = None
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
        item_16 = (
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_scaling = None
        attn_output_32 = torch._C._nn.scaled_dot_product_attention(
            query_8,
            key_17,
            value_17,
            attn_mask=attention_mask_9,
            dropout_p=0.0,
            scale=item_16,
            is_causal=False,
        )
        query_8 = key_17 = value_17 = attention_mask_9 = item_16 = None
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
        hidden_states_118 = attn_output_35.to(torch.float32)
        attn_output_35 = None
        pow_35 = hidden_states_118.pow(2)
        variance_34 = pow_35.mean(-1, keepdim=True)
        pow_35 = None
        item_17 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_68 = variance_34 + item_17
        variance_34 = item_17 = None
        rsqrt_34 = torch.rsqrt(add_68)
        add_68 = None
        hidden_states_119 = hidden_states_118 * rsqrt_34
        hidden_states_118 = rsqrt_34 = None
        to_73 = hidden_states_119.to(torch.float16)
        hidden_states_119 = None
        hidden_states_120 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
            * to_73
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = (
            to_73
        ) = None
        hidden_states_121 = hidden_states_111 + hidden_states_120
        hidden_states_111 = hidden_states_120 = None
        linear_60 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_60, inplace=False)
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_121,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_116 = silu_8 * linear_61
        silu_8 = linear_61 = None
        down_proj_8 = torch._C._nn.linear(
            mul_116,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_116 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_122 = down_proj_8.to(torch.float32)
        down_proj_8 = None
        pow_36 = hidden_states_122.pow(2)
        variance_35 = pow_36.mean(-1, keepdim=True)
        pow_36 = None
        add_70 = variance_35 + 1e-05
        variance_35 = None
        rsqrt_35 = torch.rsqrt(add_70)
        add_70 = None
        hidden_states_123 = hidden_states_122 * rsqrt_35
        hidden_states_122 = rsqrt_35 = None
        to_75 = hidden_states_123.to(torch.float16)
        hidden_states_123 = None
        hidden_states_124 = (
            l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_
            * to_75
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_75
        ) = None
        hidden_states_125 = hidden_states_121 + hidden_states_124
        hidden_states_121 = hidden_states_124 = None
        linear_63 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_27 = linear_63.view((1, 19, -1, 64))
        linear_63 = None
        query_states_18 = view_27.transpose(1, 2)
        view_27 = None
        linear_64 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_28 = linear_64.view((1, 19, -1, 64))
        linear_64 = None
        key_states_18 = view_28.transpose(1, 2)
        view_28 = None
        linear_65 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_29 = linear_65.view((1, 19, -1, 64))
        linear_65 = None
        value_states_9 = view_29.transpose(1, 2)
        view_29 = None
        hidden_states_126 = query_states_18.to(torch.float32)
        query_states_18 = None
        pow_37 = hidden_states_126.pow(2)
        variance_36 = pow_37.mean(-1, keepdim=True)
        pow_37 = None
        add_72 = variance_36 + 1e-05
        variance_36 = None
        rsqrt_36 = torch.rsqrt(add_72)
        add_72 = None
        hidden_states_127 = hidden_states_126 * rsqrt_36
        hidden_states_126 = rsqrt_36 = None
        to_77 = hidden_states_127.to(torch.float16)
        hidden_states_127 = None
        query_states_19 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_
            * to_77
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_77
        ) = None
        hidden_states_128 = key_states_18.to(torch.float32)
        key_states_18 = None
        pow_38 = hidden_states_128.pow(2)
        variance_37 = pow_38.mean(-1, keepdim=True)
        pow_38 = None
        add_73 = variance_37 + 1e-05
        variance_37 = None
        rsqrt_37 = torch.rsqrt(add_73)
        add_73 = None
        hidden_states_129 = hidden_states_128 * rsqrt_37
        hidden_states_128 = rsqrt_37 = None
        to_79 = hidden_states_129.to(torch.float16)
        hidden_states_129 = None
        key_states_19 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_
            * to_79
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_79
        ) = None
        cos_12 = cos_2.unsqueeze(1)
        sin_12 = sin_2.unsqueeze(1)
        mul_123 = query_states_19 * cos_12
        x1_18 = query_states_19[(Ellipsis, slice(None, 32, None))]
        x2_18 = query_states_19[(Ellipsis, slice(32, None, None))]
        query_states_19 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_19 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_124 = cat_19 * sin_12
        cat_19 = None
        q_embed_9 = mul_123 + mul_124
        mul_123 = mul_124 = None
        mul_125 = key_states_19 * cos_12
        cos_12 = None
        x1_19 = key_states_19[(Ellipsis, slice(None, 32, None))]
        x2_19 = key_states_19[(Ellipsis, slice(32, None, None))]
        key_states_19 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_20 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_126 = cat_20 * sin_12
        cat_20 = sin_12 = None
        k_embed_9 = mul_125 + mul_126
        mul_125 = mul_126 = None
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
        hidden_states_130 = getitem_69.expand(1, 8, 4, 19, 64)
        getitem_69 = None
        key_18 = hidden_states_130.reshape(1, 32, 19, 64)
        hidden_states_130 = None
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
        hidden_states_131 = getitem_70.expand(1, 8, 4, 19, 64)
        getitem_70 = None
        value_18 = hidden_states_131.reshape(1, 32, 19, 64)
        hidden_states_131 = None
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
        item_18 = (
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_scaling = None
        attn_output_36 = torch._C._nn.scaled_dot_product_attention(
            query_9,
            key_19,
            value_19,
            attn_mask=attention_mask_10,
            dropout_p=0.0,
            scale=item_18,
            is_causal=False,
        )
        query_9 = key_19 = value_19 = attention_mask_10 = item_18 = None
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
        hidden_states_132 = attn_output_39.to(torch.float32)
        attn_output_39 = None
        pow_39 = hidden_states_132.pow(2)
        variance_38 = pow_39.mean(-1, keepdim=True)
        pow_39 = None
        item_19 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_76 = variance_38 + item_19
        variance_38 = item_19 = None
        rsqrt_38 = torch.rsqrt(add_76)
        add_76 = None
        hidden_states_133 = hidden_states_132 * rsqrt_38
        hidden_states_132 = rsqrt_38 = None
        to_81 = hidden_states_133.to(torch.float16)
        hidden_states_133 = None
        hidden_states_134 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
            * to_81
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = (
            to_81
        ) = None
        hidden_states_135 = hidden_states_125 + hidden_states_134
        hidden_states_125 = hidden_states_134 = None
        linear_67 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_67, inplace=False)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_129 = silu_9 * linear_68
        silu_9 = linear_68 = None
        down_proj_9 = torch._C._nn.linear(
            mul_129,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_129 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_136 = down_proj_9.to(torch.float32)
        down_proj_9 = None
        pow_40 = hidden_states_136.pow(2)
        variance_39 = pow_40.mean(-1, keepdim=True)
        pow_40 = None
        add_78 = variance_39 + 1e-05
        variance_39 = None
        rsqrt_39 = torch.rsqrt(add_78)
        add_78 = None
        hidden_states_137 = hidden_states_136 * rsqrt_39
        hidden_states_136 = rsqrt_39 = None
        to_83 = hidden_states_137.to(torch.float16)
        hidden_states_137 = None
        hidden_states_138 = (
            l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_
            * to_83
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_83
        ) = None
        hidden_states_139 = hidden_states_135 + hidden_states_138
        hidden_states_135 = hidden_states_138 = None
        linear_70 = torch._C._nn.linear(
            hidden_states_139,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_30 = linear_70.view((1, 19, -1, 64))
        linear_70 = None
        query_states_20 = view_30.transpose(1, 2)
        view_30 = None
        linear_71 = torch._C._nn.linear(
            hidden_states_139,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_31 = linear_71.view((1, 19, -1, 64))
        linear_71 = None
        key_states_20 = view_31.transpose(1, 2)
        view_31 = None
        linear_72 = torch._C._nn.linear(
            hidden_states_139,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_32 = linear_72.view((1, 19, -1, 64))
        linear_72 = None
        value_states_10 = view_32.transpose(1, 2)
        view_32 = None
        hidden_states_140 = query_states_20.to(torch.float32)
        query_states_20 = None
        pow_41 = hidden_states_140.pow(2)
        variance_40 = pow_41.mean(-1, keepdim=True)
        pow_41 = None
        add_80 = variance_40 + 1e-05
        variance_40 = None
        rsqrt_40 = torch.rsqrt(add_80)
        add_80 = None
        hidden_states_141 = hidden_states_140 * rsqrt_40
        hidden_states_140 = rsqrt_40 = None
        to_85 = hidden_states_141.to(torch.float16)
        hidden_states_141 = None
        query_states_21 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_
            * to_85
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_85
        ) = None
        hidden_states_142 = key_states_20.to(torch.float32)
        key_states_20 = None
        pow_42 = hidden_states_142.pow(2)
        variance_41 = pow_42.mean(-1, keepdim=True)
        pow_42 = None
        add_81 = variance_41 + 1e-05
        variance_41 = None
        rsqrt_41 = torch.rsqrt(add_81)
        add_81 = None
        hidden_states_143 = hidden_states_142 * rsqrt_41
        hidden_states_142 = rsqrt_41 = None
        to_87 = hidden_states_143.to(torch.float16)
        hidden_states_143 = None
        key_states_21 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_
            * to_87
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_87
        ) = None
        cos_13 = cos_2.unsqueeze(1)
        sin_13 = sin_2.unsqueeze(1)
        mul_136 = query_states_21 * cos_13
        x1_20 = query_states_21[(Ellipsis, slice(None, 32, None))]
        x2_20 = query_states_21[(Ellipsis, slice(32, None, None))]
        query_states_21 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_21 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_137 = cat_21 * sin_13
        cat_21 = None
        q_embed_10 = mul_136 + mul_137
        mul_136 = mul_137 = None
        mul_138 = key_states_21 * cos_13
        cos_13 = None
        x1_21 = key_states_21[(Ellipsis, slice(None, 32, None))]
        x2_21 = key_states_21[(Ellipsis, slice(32, None, None))]
        key_states_21 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_22 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_139 = cat_22 * sin_13
        cat_22 = sin_13 = None
        k_embed_10 = mul_138 + mul_139
        mul_138 = mul_139 = None
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
        hidden_states_144 = getitem_76.expand(1, 8, 4, 19, 64)
        getitem_76 = None
        key_20 = hidden_states_144.reshape(1, 32, 19, 64)
        hidden_states_144 = None
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
        hidden_states_145 = getitem_77.expand(1, 8, 4, 19, 64)
        getitem_77 = None
        value_20 = hidden_states_145.reshape(1, 32, 19, 64)
        hidden_states_145 = None
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
        item_20 = (
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_scaling = None
        attn_output_40 = torch._C._nn.scaled_dot_product_attention(
            query_10,
            key_21,
            value_21,
            attn_mask=attention_mask_11,
            dropout_p=0.0,
            scale=item_20,
            is_causal=False,
        )
        query_10 = key_21 = value_21 = attention_mask_11 = item_20 = None
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
        hidden_states_146 = attn_output_43.to(torch.float32)
        attn_output_43 = None
        pow_43 = hidden_states_146.pow(2)
        variance_42 = pow_43.mean(-1, keepdim=True)
        pow_43 = None
        item_21 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_84 = variance_42 + item_21
        variance_42 = item_21 = None
        rsqrt_42 = torch.rsqrt(add_84)
        add_84 = None
        hidden_states_147 = hidden_states_146 * rsqrt_42
        hidden_states_146 = rsqrt_42 = None
        to_89 = hidden_states_147.to(torch.float16)
        hidden_states_147 = None
        hidden_states_148 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
            * to_89
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = (
            to_89
        ) = None
        hidden_states_149 = hidden_states_139 + hidden_states_148
        hidden_states_139 = hidden_states_148 = None
        linear_74 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_149,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_142 = silu_10 * linear_75
        silu_10 = linear_75 = None
        down_proj_10 = torch._C._nn.linear(
            mul_142,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_142 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_150 = down_proj_10.to(torch.float32)
        down_proj_10 = None
        pow_44 = hidden_states_150.pow(2)
        variance_43 = pow_44.mean(-1, keepdim=True)
        pow_44 = None
        add_86 = variance_43 + 1e-05
        variance_43 = None
        rsqrt_43 = torch.rsqrt(add_86)
        add_86 = None
        hidden_states_151 = hidden_states_150 * rsqrt_43
        hidden_states_150 = rsqrt_43 = None
        to_91 = hidden_states_151.to(torch.float16)
        hidden_states_151 = None
        hidden_states_152 = (
            l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_
            * to_91
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_91
        ) = None
        hidden_states_153 = hidden_states_149 + hidden_states_152
        hidden_states_149 = hidden_states_152 = None
        linear_77 = torch._C._nn.linear(
            hidden_states_153,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_33 = linear_77.view((1, 19, -1, 64))
        linear_77 = None
        query_states_22 = view_33.transpose(1, 2)
        view_33 = None
        linear_78 = torch._C._nn.linear(
            hidden_states_153,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_34 = linear_78.view((1, 19, -1, 64))
        linear_78 = None
        key_states_22 = view_34.transpose(1, 2)
        view_34 = None
        linear_79 = torch._C._nn.linear(
            hidden_states_153,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_35 = linear_79.view((1, 19, -1, 64))
        linear_79 = None
        value_states_11 = view_35.transpose(1, 2)
        view_35 = None
        hidden_states_154 = query_states_22.to(torch.float32)
        query_states_22 = None
        pow_45 = hidden_states_154.pow(2)
        variance_44 = pow_45.mean(-1, keepdim=True)
        pow_45 = None
        add_88 = variance_44 + 1e-05
        variance_44 = None
        rsqrt_44 = torch.rsqrt(add_88)
        add_88 = None
        hidden_states_155 = hidden_states_154 * rsqrt_44
        hidden_states_154 = rsqrt_44 = None
        to_93 = hidden_states_155.to(torch.float16)
        hidden_states_155 = None
        query_states_23 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_
            * to_93
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_93
        ) = None
        hidden_states_156 = key_states_22.to(torch.float32)
        key_states_22 = None
        pow_46 = hidden_states_156.pow(2)
        variance_45 = pow_46.mean(-1, keepdim=True)
        pow_46 = None
        add_89 = variance_45 + 1e-05
        variance_45 = None
        rsqrt_45 = torch.rsqrt(add_89)
        add_89 = None
        hidden_states_157 = hidden_states_156 * rsqrt_45
        hidden_states_156 = rsqrt_45 = None
        to_95 = hidden_states_157.to(torch.float16)
        hidden_states_157 = None
        key_states_23 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_
            * to_95
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_95
        ) = None
        cos_14 = cos_2.unsqueeze(1)
        sin_14 = sin_2.unsqueeze(1)
        mul_149 = query_states_23 * cos_14
        x1_22 = query_states_23[(Ellipsis, slice(None, 32, None))]
        x2_22 = query_states_23[(Ellipsis, slice(32, None, None))]
        query_states_23 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_23 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_150 = cat_23 * sin_14
        cat_23 = None
        q_embed_11 = mul_149 + mul_150
        mul_149 = mul_150 = None
        mul_151 = key_states_23 * cos_14
        cos_14 = None
        x1_23 = key_states_23[(Ellipsis, slice(None, 32, None))]
        x2_23 = key_states_23[(Ellipsis, slice(32, None, None))]
        key_states_23 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_24 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_152 = cat_24 * sin_14
        cat_24 = sin_14 = None
        k_embed_11 = mul_151 + mul_152
        mul_151 = mul_152 = None
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
        hidden_states_158 = getitem_83.expand(1, 8, 4, 19, 64)
        getitem_83 = None
        key_22 = hidden_states_158.reshape(1, 32, 19, 64)
        hidden_states_158 = None
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
        hidden_states_159 = getitem_84.expand(1, 8, 4, 19, 64)
        getitem_84 = None
        value_22 = hidden_states_159.reshape(1, 32, 19, 64)
        hidden_states_159 = None
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
        item_22 = (
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_scaling = None
        attn_output_44 = torch._C._nn.scaled_dot_product_attention(
            query_11,
            key_23,
            value_23,
            attn_mask=attention_mask_12,
            dropout_p=0.0,
            scale=item_22,
            is_causal=False,
        )
        query_11 = key_23 = value_23 = attention_mask_12 = item_22 = None
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
        hidden_states_160 = attn_output_47.to(torch.float32)
        attn_output_47 = None
        pow_47 = hidden_states_160.pow(2)
        variance_46 = pow_47.mean(-1, keepdim=True)
        pow_47 = None
        item_23 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_92 = variance_46 + item_23
        variance_46 = item_23 = None
        rsqrt_46 = torch.rsqrt(add_92)
        add_92 = None
        hidden_states_161 = hidden_states_160 * rsqrt_46
        hidden_states_160 = rsqrt_46 = None
        to_97 = hidden_states_161.to(torch.float16)
        hidden_states_161 = None
        hidden_states_162 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
            * to_97
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = (
            to_97
        ) = None
        hidden_states_163 = hidden_states_153 + hidden_states_162
        hidden_states_153 = hidden_states_162 = None
        linear_81 = torch._C._nn.linear(
            hidden_states_163,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_81, inplace=False)
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_163,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_155 = silu_11 * linear_82
        silu_11 = linear_82 = None
        down_proj_11 = torch._C._nn.linear(
            mul_155,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_155 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_164 = down_proj_11.to(torch.float32)
        down_proj_11 = None
        pow_48 = hidden_states_164.pow(2)
        variance_47 = pow_48.mean(-1, keepdim=True)
        pow_48 = None
        add_94 = variance_47 + 1e-05
        variance_47 = None
        rsqrt_47 = torch.rsqrt(add_94)
        add_94 = None
        hidden_states_165 = hidden_states_164 * rsqrt_47
        hidden_states_164 = rsqrt_47 = None
        to_99 = hidden_states_165.to(torch.float16)
        hidden_states_165 = None
        hidden_states_166 = (
            l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_
            * to_99
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_99
        ) = None
        hidden_states_167 = hidden_states_163 + hidden_states_166
        hidden_states_163 = hidden_states_166 = None
        linear_84 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_36 = linear_84.view((1, 19, -1, 64))
        linear_84 = None
        query_states_24 = view_36.transpose(1, 2)
        view_36 = None
        linear_85 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_37 = linear_85.view((1, 19, -1, 64))
        linear_85 = None
        key_states_24 = view_37.transpose(1, 2)
        view_37 = None
        linear_86 = torch._C._nn.linear(
            hidden_states_167,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_38 = linear_86.view((1, 19, -1, 64))
        linear_86 = None
        value_states_12 = view_38.transpose(1, 2)
        view_38 = None
        hidden_states_168 = query_states_24.to(torch.float32)
        query_states_24 = None
        pow_49 = hidden_states_168.pow(2)
        variance_48 = pow_49.mean(-1, keepdim=True)
        pow_49 = None
        add_96 = variance_48 + 1e-05
        variance_48 = None
        rsqrt_48 = torch.rsqrt(add_96)
        add_96 = None
        hidden_states_169 = hidden_states_168 * rsqrt_48
        hidden_states_168 = rsqrt_48 = None
        to_101 = hidden_states_169.to(torch.float16)
        hidden_states_169 = None
        query_states_25 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_
            * to_101
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_101
        ) = None
        hidden_states_170 = key_states_24.to(torch.float32)
        key_states_24 = None
        pow_50 = hidden_states_170.pow(2)
        variance_49 = pow_50.mean(-1, keepdim=True)
        pow_50 = None
        add_97 = variance_49 + 1e-05
        variance_49 = None
        rsqrt_49 = torch.rsqrt(add_97)
        add_97 = None
        hidden_states_171 = hidden_states_170 * rsqrt_49
        hidden_states_170 = rsqrt_49 = None
        to_103 = hidden_states_171.to(torch.float16)
        hidden_states_171 = None
        key_states_25 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_
            * to_103
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_103
        ) = None
        cos_15 = cos_2.unsqueeze(1)
        sin_15 = sin_2.unsqueeze(1)
        mul_162 = query_states_25 * cos_15
        x1_24 = query_states_25[(Ellipsis, slice(None, 32, None))]
        x2_24 = query_states_25[(Ellipsis, slice(32, None, None))]
        query_states_25 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_25 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_163 = cat_25 * sin_15
        cat_25 = None
        q_embed_12 = mul_162 + mul_163
        mul_162 = mul_163 = None
        mul_164 = key_states_25 * cos_15
        cos_15 = None
        x1_25 = key_states_25[(Ellipsis, slice(None, 32, None))]
        x2_25 = key_states_25[(Ellipsis, slice(32, None, None))]
        key_states_25 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_26 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_165 = cat_26 * sin_15
        cat_26 = sin_15 = None
        k_embed_12 = mul_164 + mul_165
        mul_164 = mul_165 = None
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
        hidden_states_172 = getitem_90.expand(1, 8, 4, 19, 64)
        getitem_90 = None
        key_24 = hidden_states_172.reshape(1, 32, 19, 64)
        hidden_states_172 = None
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
        hidden_states_173 = getitem_91.expand(1, 8, 4, 19, 64)
        getitem_91 = None
        value_24 = hidden_states_173.reshape(1, 32, 19, 64)
        hidden_states_173 = None
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
        item_24 = (
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_scaling = None
        attn_output_48 = torch._C._nn.scaled_dot_product_attention(
            query_12,
            key_25,
            value_25,
            attn_mask=attention_mask_13,
            dropout_p=0.0,
            scale=item_24,
            is_causal=False,
        )
        query_12 = key_25 = value_25 = attention_mask_13 = item_24 = None
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
        hidden_states_174 = attn_output_51.to(torch.float32)
        attn_output_51 = None
        pow_51 = hidden_states_174.pow(2)
        variance_50 = pow_51.mean(-1, keepdim=True)
        pow_51 = None
        item_25 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_100 = variance_50 + item_25
        variance_50 = item_25 = None
        rsqrt_50 = torch.rsqrt(add_100)
        add_100 = None
        hidden_states_175 = hidden_states_174 * rsqrt_50
        hidden_states_174 = rsqrt_50 = None
        to_105 = hidden_states_175.to(torch.float16)
        hidden_states_175 = None
        hidden_states_176 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
            * to_105
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = (
            to_105
        ) = None
        hidden_states_177 = hidden_states_167 + hidden_states_176
        hidden_states_167 = hidden_states_176 = None
        linear_88 = torch._C._nn.linear(
            hidden_states_177,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_88, inplace=False)
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_177,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_168 = silu_12 * linear_89
        silu_12 = linear_89 = None
        down_proj_12 = torch._C._nn.linear(
            mul_168,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_168 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_178 = down_proj_12.to(torch.float32)
        down_proj_12 = None
        pow_52 = hidden_states_178.pow(2)
        variance_51 = pow_52.mean(-1, keepdim=True)
        pow_52 = None
        add_102 = variance_51 + 1e-05
        variance_51 = None
        rsqrt_51 = torch.rsqrt(add_102)
        add_102 = None
        hidden_states_179 = hidden_states_178 * rsqrt_51
        hidden_states_178 = rsqrt_51 = None
        to_107 = hidden_states_179.to(torch.float16)
        hidden_states_179 = None
        hidden_states_180 = (
            l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_
            * to_107
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_107
        ) = None
        hidden_states_181 = hidden_states_177 + hidden_states_180
        hidden_states_177 = hidden_states_180 = None
        linear_91 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_39 = linear_91.view((1, 19, -1, 64))
        linear_91 = None
        query_states_26 = view_39.transpose(1, 2)
        view_39 = None
        linear_92 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_40 = linear_92.view((1, 19, -1, 64))
        linear_92 = None
        key_states_26 = view_40.transpose(1, 2)
        view_40 = None
        linear_93 = torch._C._nn.linear(
            hidden_states_181,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_41 = linear_93.view((1, 19, -1, 64))
        linear_93 = None
        value_states_13 = view_41.transpose(1, 2)
        view_41 = None
        hidden_states_182 = query_states_26.to(torch.float32)
        query_states_26 = None
        pow_53 = hidden_states_182.pow(2)
        variance_52 = pow_53.mean(-1, keepdim=True)
        pow_53 = None
        add_104 = variance_52 + 1e-05
        variance_52 = None
        rsqrt_52 = torch.rsqrt(add_104)
        add_104 = None
        hidden_states_183 = hidden_states_182 * rsqrt_52
        hidden_states_182 = rsqrt_52 = None
        to_109 = hidden_states_183.to(torch.float16)
        hidden_states_183 = None
        query_states_27 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_
            * to_109
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_109
        ) = None
        hidden_states_184 = key_states_26.to(torch.float32)
        key_states_26 = None
        pow_54 = hidden_states_184.pow(2)
        variance_53 = pow_54.mean(-1, keepdim=True)
        pow_54 = None
        add_105 = variance_53 + 1e-05
        variance_53 = None
        rsqrt_53 = torch.rsqrt(add_105)
        add_105 = None
        hidden_states_185 = hidden_states_184 * rsqrt_53
        hidden_states_184 = rsqrt_53 = None
        to_111 = hidden_states_185.to(torch.float16)
        hidden_states_185 = None
        key_states_27 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_
            * to_111
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_111
        ) = None
        cos_16 = cos_2.unsqueeze(1)
        sin_16 = sin_2.unsqueeze(1)
        mul_175 = query_states_27 * cos_16
        x1_26 = query_states_27[(Ellipsis, slice(None, 32, None))]
        x2_26 = query_states_27[(Ellipsis, slice(32, None, None))]
        query_states_27 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_27 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_176 = cat_27 * sin_16
        cat_27 = None
        q_embed_13 = mul_175 + mul_176
        mul_175 = mul_176 = None
        mul_177 = key_states_27 * cos_16
        cos_16 = None
        x1_27 = key_states_27[(Ellipsis, slice(None, 32, None))]
        x2_27 = key_states_27[(Ellipsis, slice(32, None, None))]
        key_states_27 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_28 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_178 = cat_28 * sin_16
        cat_28 = sin_16 = None
        k_embed_13 = mul_177 + mul_178
        mul_177 = mul_178 = None
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
        hidden_states_186 = getitem_97.expand(1, 8, 4, 19, 64)
        getitem_97 = None
        key_26 = hidden_states_186.reshape(1, 32, 19, 64)
        hidden_states_186 = None
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
        hidden_states_187 = getitem_98.expand(1, 8, 4, 19, 64)
        getitem_98 = None
        value_26 = hidden_states_187.reshape(1, 32, 19, 64)
        hidden_states_187 = None
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
        item_26 = (
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_scaling = None
        attn_output_52 = torch._C._nn.scaled_dot_product_attention(
            query_13,
            key_27,
            value_27,
            attn_mask=attention_mask_14,
            dropout_p=0.0,
            scale=item_26,
            is_causal=False,
        )
        query_13 = key_27 = value_27 = attention_mask_14 = item_26 = None
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
        hidden_states_188 = attn_output_55.to(torch.float32)
        attn_output_55 = None
        pow_55 = hidden_states_188.pow(2)
        variance_54 = pow_55.mean(-1, keepdim=True)
        pow_55 = None
        item_27 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_108 = variance_54 + item_27
        variance_54 = item_27 = None
        rsqrt_54 = torch.rsqrt(add_108)
        add_108 = None
        hidden_states_189 = hidden_states_188 * rsqrt_54
        hidden_states_188 = rsqrt_54 = None
        to_113 = hidden_states_189.to(torch.float16)
        hidden_states_189 = None
        hidden_states_190 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
            * to_113
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = (
            to_113
        ) = None
        hidden_states_191 = hidden_states_181 + hidden_states_190
        hidden_states_181 = hidden_states_190 = None
        linear_95 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_95, inplace=False)
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_191,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_181 = silu_13 * linear_96
        silu_13 = linear_96 = None
        down_proj_13 = torch._C._nn.linear(
            mul_181,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_181 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_192 = down_proj_13.to(torch.float32)
        down_proj_13 = None
        pow_56 = hidden_states_192.pow(2)
        variance_55 = pow_56.mean(-1, keepdim=True)
        pow_56 = None
        add_110 = variance_55 + 1e-05
        variance_55 = None
        rsqrt_55 = torch.rsqrt(add_110)
        add_110 = None
        hidden_states_193 = hidden_states_192 * rsqrt_55
        hidden_states_192 = rsqrt_55 = None
        to_115 = hidden_states_193.to(torch.float16)
        hidden_states_193 = None
        hidden_states_194 = (
            l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_
            * to_115
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_115
        ) = None
        hidden_states_195 = hidden_states_191 + hidden_states_194
        hidden_states_191 = hidden_states_194 = None
        linear_98 = torch._C._nn.linear(
            hidden_states_195,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_42 = linear_98.view((1, 19, -1, 64))
        linear_98 = None
        query_states_28 = view_42.transpose(1, 2)
        view_42 = None
        linear_99 = torch._C._nn.linear(
            hidden_states_195,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_43 = linear_99.view((1, 19, -1, 64))
        linear_99 = None
        key_states_28 = view_43.transpose(1, 2)
        view_43 = None
        linear_100 = torch._C._nn.linear(
            hidden_states_195,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_44 = linear_100.view((1, 19, -1, 64))
        linear_100 = None
        value_states_14 = view_44.transpose(1, 2)
        view_44 = None
        hidden_states_196 = query_states_28.to(torch.float32)
        query_states_28 = None
        pow_57 = hidden_states_196.pow(2)
        variance_56 = pow_57.mean(-1, keepdim=True)
        pow_57 = None
        add_112 = variance_56 + 1e-05
        variance_56 = None
        rsqrt_56 = torch.rsqrt(add_112)
        add_112 = None
        hidden_states_197 = hidden_states_196 * rsqrt_56
        hidden_states_196 = rsqrt_56 = None
        to_117 = hidden_states_197.to(torch.float16)
        hidden_states_197 = None
        query_states_29 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_
            * to_117
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_117
        ) = None
        hidden_states_198 = key_states_28.to(torch.float32)
        key_states_28 = None
        pow_58 = hidden_states_198.pow(2)
        variance_57 = pow_58.mean(-1, keepdim=True)
        pow_58 = None
        add_113 = variance_57 + 1e-05
        variance_57 = None
        rsqrt_57 = torch.rsqrt(add_113)
        add_113 = None
        hidden_states_199 = hidden_states_198 * rsqrt_57
        hidden_states_198 = rsqrt_57 = None
        to_119 = hidden_states_199.to(torch.float16)
        hidden_states_199 = None
        key_states_29 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_
            * to_119
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_119
        ) = None
        cos_17 = cos_2.unsqueeze(1)
        sin_17 = sin_2.unsqueeze(1)
        mul_188 = query_states_29 * cos_17
        x1_28 = query_states_29[(Ellipsis, slice(None, 32, None))]
        x2_28 = query_states_29[(Ellipsis, slice(32, None, None))]
        query_states_29 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_29 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_189 = cat_29 * sin_17
        cat_29 = None
        q_embed_14 = mul_188 + mul_189
        mul_188 = mul_189 = None
        mul_190 = key_states_29 * cos_17
        cos_17 = None
        x1_29 = key_states_29[(Ellipsis, slice(None, 32, None))]
        x2_29 = key_states_29[(Ellipsis, slice(32, None, None))]
        key_states_29 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_30 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_191 = cat_30 * sin_17
        cat_30 = sin_17 = None
        k_embed_14 = mul_190 + mul_191
        mul_190 = mul_191 = None
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
        hidden_states_200 = getitem_104.expand(1, 8, 4, 19, 64)
        getitem_104 = None
        key_28 = hidden_states_200.reshape(1, 32, 19, 64)
        hidden_states_200 = None
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
        hidden_states_201 = getitem_105.expand(1, 8, 4, 19, 64)
        getitem_105 = None
        value_28 = hidden_states_201.reshape(1, 32, 19, 64)
        hidden_states_201 = None
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
        item_28 = (
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_scaling = None
        attn_output_56 = torch._C._nn.scaled_dot_product_attention(
            query_14,
            key_29,
            value_29,
            attn_mask=attention_mask_15,
            dropout_p=0.0,
            scale=item_28,
            is_causal=False,
        )
        query_14 = key_29 = value_29 = attention_mask_15 = item_28 = None
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
        hidden_states_202 = attn_output_59.to(torch.float32)
        attn_output_59 = None
        pow_59 = hidden_states_202.pow(2)
        variance_58 = pow_59.mean(-1, keepdim=True)
        pow_59 = None
        item_29 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_116 = variance_58 + item_29
        variance_58 = item_29 = None
        rsqrt_58 = torch.rsqrt(add_116)
        add_116 = None
        hidden_states_203 = hidden_states_202 * rsqrt_58
        hidden_states_202 = rsqrt_58 = None
        to_121 = hidden_states_203.to(torch.float16)
        hidden_states_203 = None
        hidden_states_204 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
            * to_121
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = (
            to_121
        ) = None
        hidden_states_205 = hidden_states_195 + hidden_states_204
        hidden_states_195 = hidden_states_204 = None
        linear_102 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_102, inplace=False)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_205,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_194 = silu_14 * linear_103
        silu_14 = linear_103 = None
        down_proj_14 = torch._C._nn.linear(
            mul_194,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_194 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_206 = down_proj_14.to(torch.float32)
        down_proj_14 = None
        pow_60 = hidden_states_206.pow(2)
        variance_59 = pow_60.mean(-1, keepdim=True)
        pow_60 = None
        add_118 = variance_59 + 1e-05
        variance_59 = None
        rsqrt_59 = torch.rsqrt(add_118)
        add_118 = None
        hidden_states_207 = hidden_states_206 * rsqrt_59
        hidden_states_206 = rsqrt_59 = None
        to_123 = hidden_states_207.to(torch.float16)
        hidden_states_207 = None
        hidden_states_208 = (
            l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_
            * to_123
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_123
        ) = None
        hidden_states_209 = hidden_states_205 + hidden_states_208
        hidden_states_205 = hidden_states_208 = None
        linear_105 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_45 = linear_105.view((1, 19, -1, 64))
        linear_105 = None
        query_states_30 = view_45.transpose(1, 2)
        view_45 = None
        linear_106 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_46 = linear_106.view((1, 19, -1, 64))
        linear_106 = None
        key_states_30 = view_46.transpose(1, 2)
        view_46 = None
        linear_107 = torch._C._nn.linear(
            hidden_states_209,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_47 = linear_107.view((1, 19, -1, 64))
        linear_107 = None
        value_states_15 = view_47.transpose(1, 2)
        view_47 = None
        hidden_states_210 = query_states_30.to(torch.float32)
        query_states_30 = None
        pow_61 = hidden_states_210.pow(2)
        variance_60 = pow_61.mean(-1, keepdim=True)
        pow_61 = None
        add_120 = variance_60 + 1e-05
        variance_60 = None
        rsqrt_60 = torch.rsqrt(add_120)
        add_120 = None
        hidden_states_211 = hidden_states_210 * rsqrt_60
        hidden_states_210 = rsqrt_60 = None
        to_125 = hidden_states_211.to(torch.float16)
        hidden_states_211 = None
        query_states_31 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_
            * to_125
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_125
        ) = None
        hidden_states_212 = key_states_30.to(torch.float32)
        key_states_30 = None
        pow_62 = hidden_states_212.pow(2)
        variance_61 = pow_62.mean(-1, keepdim=True)
        pow_62 = None
        add_121 = variance_61 + 1e-05
        variance_61 = None
        rsqrt_61 = torch.rsqrt(add_121)
        add_121 = None
        hidden_states_213 = hidden_states_212 * rsqrt_61
        hidden_states_212 = rsqrt_61 = None
        to_127 = hidden_states_213.to(torch.float16)
        hidden_states_213 = None
        key_states_31 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_
            * to_127
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_127
        ) = None
        cos_18 = cos_2.unsqueeze(1)
        sin_18 = sin_2.unsqueeze(1)
        mul_201 = query_states_31 * cos_18
        x1_30 = query_states_31[(Ellipsis, slice(None, 32, None))]
        x2_30 = query_states_31[(Ellipsis, slice(32, None, None))]
        query_states_31 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_31 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_202 = cat_31 * sin_18
        cat_31 = None
        q_embed_15 = mul_201 + mul_202
        mul_201 = mul_202 = None
        mul_203 = key_states_31 * cos_18
        cos_18 = None
        x1_31 = key_states_31[(Ellipsis, slice(None, 32, None))]
        x2_31 = key_states_31[(Ellipsis, slice(32, None, None))]
        key_states_31 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_32 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_204 = cat_32 * sin_18
        cat_32 = sin_18 = None
        k_embed_15 = mul_203 + mul_204
        mul_203 = mul_204 = None
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
        hidden_states_214 = getitem_111.expand(1, 8, 4, 19, 64)
        getitem_111 = None
        key_30 = hidden_states_214.reshape(1, 32, 19, 64)
        hidden_states_214 = None
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
        hidden_states_215 = getitem_112.expand(1, 8, 4, 19, 64)
        getitem_112 = None
        value_30 = hidden_states_215.reshape(1, 32, 19, 64)
        hidden_states_215 = None
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
        item_30 = (
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_scaling = None
        attn_output_60 = torch._C._nn.scaled_dot_product_attention(
            query_15,
            key_31,
            value_31,
            attn_mask=attention_mask_16,
            dropout_p=0.0,
            scale=item_30,
            is_causal=False,
        )
        query_15 = key_31 = value_31 = attention_mask_16 = item_30 = None
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
        hidden_states_216 = attn_output_63.to(torch.float32)
        attn_output_63 = None
        pow_63 = hidden_states_216.pow(2)
        variance_62 = pow_63.mean(-1, keepdim=True)
        pow_63 = None
        item_31 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_124 = variance_62 + item_31
        variance_62 = item_31 = None
        rsqrt_62 = torch.rsqrt(add_124)
        add_124 = None
        hidden_states_217 = hidden_states_216 * rsqrt_62
        hidden_states_216 = rsqrt_62 = None
        to_129 = hidden_states_217.to(torch.float16)
        hidden_states_217 = None
        hidden_states_218 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
            * to_129
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = (
            to_129
        ) = None
        hidden_states_219 = hidden_states_209 + hidden_states_218
        hidden_states_209 = hidden_states_218 = None
        linear_109 = torch._C._nn.linear(
            hidden_states_219,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_109, inplace=False)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_219,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_207 = silu_15 * linear_110
        silu_15 = linear_110 = None
        down_proj_15 = torch._C._nn.linear(
            mul_207,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_207 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_220 = down_proj_15.to(torch.float32)
        down_proj_15 = None
        pow_64 = hidden_states_220.pow(2)
        variance_63 = pow_64.mean(-1, keepdim=True)
        pow_64 = None
        add_126 = variance_63 + 1e-05
        variance_63 = None
        rsqrt_63 = torch.rsqrt(add_126)
        add_126 = None
        hidden_states_221 = hidden_states_220 * rsqrt_63
        hidden_states_220 = rsqrt_63 = None
        to_131 = hidden_states_221.to(torch.float16)
        hidden_states_221 = None
        hidden_states_222 = (
            l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_
            * to_131
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_131
        ) = None
        hidden_states_223 = hidden_states_219 + hidden_states_222
        hidden_states_219 = hidden_states_222 = None
        linear_112 = torch._C._nn.linear(
            hidden_states_223,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_48 = linear_112.view((1, 19, -1, 64))
        linear_112 = None
        query_states_32 = view_48.transpose(1, 2)
        view_48 = None
        linear_113 = torch._C._nn.linear(
            hidden_states_223,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_49 = linear_113.view((1, 19, -1, 64))
        linear_113 = None
        key_states_32 = view_49.transpose(1, 2)
        view_49 = None
        linear_114 = torch._C._nn.linear(
            hidden_states_223,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_50 = linear_114.view((1, 19, -1, 64))
        linear_114 = None
        value_states_16 = view_50.transpose(1, 2)
        view_50 = None
        hidden_states_224 = query_states_32.to(torch.float32)
        query_states_32 = None
        pow_65 = hidden_states_224.pow(2)
        variance_64 = pow_65.mean(-1, keepdim=True)
        pow_65 = None
        add_128 = variance_64 + 1e-05
        variance_64 = None
        rsqrt_64 = torch.rsqrt(add_128)
        add_128 = None
        hidden_states_225 = hidden_states_224 * rsqrt_64
        hidden_states_224 = rsqrt_64 = None
        to_133 = hidden_states_225.to(torch.float16)
        hidden_states_225 = None
        query_states_33 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_
            * to_133
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_133
        ) = None
        hidden_states_226 = key_states_32.to(torch.float32)
        key_states_32 = None
        pow_66 = hidden_states_226.pow(2)
        variance_65 = pow_66.mean(-1, keepdim=True)
        pow_66 = None
        add_129 = variance_65 + 1e-05
        variance_65 = None
        rsqrt_65 = torch.rsqrt(add_129)
        add_129 = None
        hidden_states_227 = hidden_states_226 * rsqrt_65
        hidden_states_226 = rsqrt_65 = None
        to_135 = hidden_states_227.to(torch.float16)
        hidden_states_227 = None
        key_states_33 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_
            * to_135
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_135
        ) = None
        cos_19 = cos_2.unsqueeze(1)
        sin_19 = sin_2.unsqueeze(1)
        mul_214 = query_states_33 * cos_19
        x1_32 = query_states_33[(Ellipsis, slice(None, 32, None))]
        x2_32 = query_states_33[(Ellipsis, slice(32, None, None))]
        query_states_33 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_33 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_215 = cat_33 * sin_19
        cat_33 = None
        q_embed_16 = mul_214 + mul_215
        mul_214 = mul_215 = None
        mul_216 = key_states_33 * cos_19
        cos_19 = None
        x1_33 = key_states_33[(Ellipsis, slice(None, 32, None))]
        x2_33 = key_states_33[(Ellipsis, slice(32, None, None))]
        key_states_33 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_34 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_217 = cat_34 * sin_19
        cat_34 = sin_19 = None
        k_embed_16 = mul_216 + mul_217
        mul_216 = mul_217 = None
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
        hidden_states_228 = getitem_118.expand(1, 8, 4, 19, 64)
        getitem_118 = None
        key_32 = hidden_states_228.reshape(1, 32, 19, 64)
        hidden_states_228 = None
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
        hidden_states_229 = getitem_119.expand(1, 8, 4, 19, 64)
        getitem_119 = None
        value_32 = hidden_states_229.reshape(1, 32, 19, 64)
        hidden_states_229 = None
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
        item_32 = (
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_scaling = None
        attn_output_64 = torch._C._nn.scaled_dot_product_attention(
            query_16,
            key_33,
            value_33,
            attn_mask=attention_mask_17,
            dropout_p=0.0,
            scale=item_32,
            is_causal=False,
        )
        query_16 = key_33 = value_33 = attention_mask_17 = item_32 = None
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
        hidden_states_230 = attn_output_67.to(torch.float32)
        attn_output_67 = None
        pow_67 = hidden_states_230.pow(2)
        variance_66 = pow_67.mean(-1, keepdim=True)
        pow_67 = None
        item_33 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_132 = variance_66 + item_33
        variance_66 = item_33 = None
        rsqrt_66 = torch.rsqrt(add_132)
        add_132 = None
        hidden_states_231 = hidden_states_230 * rsqrt_66
        hidden_states_230 = rsqrt_66 = None
        to_137 = hidden_states_231.to(torch.float16)
        hidden_states_231 = None
        hidden_states_232 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
            * to_137
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = (
            to_137
        ) = None
        hidden_states_233 = hidden_states_223 + hidden_states_232
        hidden_states_223 = hidden_states_232 = None
        linear_116 = torch._C._nn.linear(
            hidden_states_233,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_116, inplace=False)
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_233,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_220 = silu_16 * linear_117
        silu_16 = linear_117 = None
        down_proj_16 = torch._C._nn.linear(
            mul_220,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_220 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_234 = down_proj_16.to(torch.float32)
        down_proj_16 = None
        pow_68 = hidden_states_234.pow(2)
        variance_67 = pow_68.mean(-1, keepdim=True)
        pow_68 = None
        add_134 = variance_67 + 1e-05
        variance_67 = None
        rsqrt_67 = torch.rsqrt(add_134)
        add_134 = None
        hidden_states_235 = hidden_states_234 * rsqrt_67
        hidden_states_234 = rsqrt_67 = None
        to_139 = hidden_states_235.to(torch.float16)
        hidden_states_235 = None
        hidden_states_236 = (
            l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_
            * to_139
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_139
        ) = None
        hidden_states_237 = hidden_states_233 + hidden_states_236
        hidden_states_233 = hidden_states_236 = None
        linear_119 = torch._C._nn.linear(
            hidden_states_237,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_51 = linear_119.view((1, 19, -1, 64))
        linear_119 = None
        query_states_34 = view_51.transpose(1, 2)
        view_51 = None
        linear_120 = torch._C._nn.linear(
            hidden_states_237,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_52 = linear_120.view((1, 19, -1, 64))
        linear_120 = None
        key_states_34 = view_52.transpose(1, 2)
        view_52 = None
        linear_121 = torch._C._nn.linear(
            hidden_states_237,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_53 = linear_121.view((1, 19, -1, 64))
        linear_121 = None
        value_states_17 = view_53.transpose(1, 2)
        view_53 = None
        hidden_states_238 = query_states_34.to(torch.float32)
        query_states_34 = None
        pow_69 = hidden_states_238.pow(2)
        variance_68 = pow_69.mean(-1, keepdim=True)
        pow_69 = None
        add_136 = variance_68 + 1e-05
        variance_68 = None
        rsqrt_68 = torch.rsqrt(add_136)
        add_136 = None
        hidden_states_239 = hidden_states_238 * rsqrt_68
        hidden_states_238 = rsqrt_68 = None
        to_141 = hidden_states_239.to(torch.float16)
        hidden_states_239 = None
        query_states_35 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_
            * to_141
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_141
        ) = None
        hidden_states_240 = key_states_34.to(torch.float32)
        key_states_34 = None
        pow_70 = hidden_states_240.pow(2)
        variance_69 = pow_70.mean(-1, keepdim=True)
        pow_70 = None
        add_137 = variance_69 + 1e-05
        variance_69 = None
        rsqrt_69 = torch.rsqrt(add_137)
        add_137 = None
        hidden_states_241 = hidden_states_240 * rsqrt_69
        hidden_states_240 = rsqrt_69 = None
        to_143 = hidden_states_241.to(torch.float16)
        hidden_states_241 = None
        key_states_35 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_
            * to_143
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_143
        ) = None
        cos_20 = cos_2.unsqueeze(1)
        sin_20 = sin_2.unsqueeze(1)
        mul_227 = query_states_35 * cos_20
        x1_34 = query_states_35[(Ellipsis, slice(None, 32, None))]
        x2_34 = query_states_35[(Ellipsis, slice(32, None, None))]
        query_states_35 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_35 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_228 = cat_35 * sin_20
        cat_35 = None
        q_embed_17 = mul_227 + mul_228
        mul_227 = mul_228 = None
        mul_229 = key_states_35 * cos_20
        cos_20 = None
        x1_35 = key_states_35[(Ellipsis, slice(None, 32, None))]
        x2_35 = key_states_35[(Ellipsis, slice(32, None, None))]
        key_states_35 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_36 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_230 = cat_36 * sin_20
        cat_36 = sin_20 = None
        k_embed_17 = mul_229 + mul_230
        mul_229 = mul_230 = None
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
        hidden_states_242 = getitem_125.expand(1, 8, 4, 19, 64)
        getitem_125 = None
        key_34 = hidden_states_242.reshape(1, 32, 19, 64)
        hidden_states_242 = None
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
        hidden_states_243 = getitem_126.expand(1, 8, 4, 19, 64)
        getitem_126 = None
        value_34 = hidden_states_243.reshape(1, 32, 19, 64)
        hidden_states_243 = None
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
        item_34 = (
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_scaling = None
        attn_output_68 = torch._C._nn.scaled_dot_product_attention(
            query_17,
            key_35,
            value_35,
            attn_mask=attention_mask_18,
            dropout_p=0.0,
            scale=item_34,
            is_causal=False,
        )
        query_17 = key_35 = value_35 = attention_mask_18 = item_34 = None
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
        hidden_states_244 = attn_output_71.to(torch.float32)
        attn_output_71 = None
        pow_71 = hidden_states_244.pow(2)
        variance_70 = pow_71.mean(-1, keepdim=True)
        pow_71 = None
        item_35 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_140 = variance_70 + item_35
        variance_70 = item_35 = None
        rsqrt_70 = torch.rsqrt(add_140)
        add_140 = None
        hidden_states_245 = hidden_states_244 * rsqrt_70
        hidden_states_244 = rsqrt_70 = None
        to_145 = hidden_states_245.to(torch.float16)
        hidden_states_245 = None
        hidden_states_246 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
            * to_145
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = (
            to_145
        ) = None
        hidden_states_247 = hidden_states_237 + hidden_states_246
        hidden_states_237 = hidden_states_246 = None
        linear_123 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_123, inplace=False)
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_247,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_233 = silu_17 * linear_124
        silu_17 = linear_124 = None
        down_proj_17 = torch._C._nn.linear(
            mul_233,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_233 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_248 = down_proj_17.to(torch.float32)
        down_proj_17 = None
        pow_72 = hidden_states_248.pow(2)
        variance_71 = pow_72.mean(-1, keepdim=True)
        pow_72 = None
        add_142 = variance_71 + 1e-05
        variance_71 = None
        rsqrt_71 = torch.rsqrt(add_142)
        add_142 = None
        hidden_states_249 = hidden_states_248 * rsqrt_71
        hidden_states_248 = rsqrt_71 = None
        to_147 = hidden_states_249.to(torch.float16)
        hidden_states_249 = None
        hidden_states_250 = (
            l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_
            * to_147
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_147
        ) = None
        hidden_states_251 = hidden_states_247 + hidden_states_250
        hidden_states_247 = hidden_states_250 = None
        linear_126 = torch._C._nn.linear(
            hidden_states_251,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_54 = linear_126.view((1, 19, -1, 64))
        linear_126 = None
        query_states_36 = view_54.transpose(1, 2)
        view_54 = None
        linear_127 = torch._C._nn.linear(
            hidden_states_251,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_55 = linear_127.view((1, 19, -1, 64))
        linear_127 = None
        key_states_36 = view_55.transpose(1, 2)
        view_55 = None
        linear_128 = torch._C._nn.linear(
            hidden_states_251,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_56 = linear_128.view((1, 19, -1, 64))
        linear_128 = None
        value_states_18 = view_56.transpose(1, 2)
        view_56 = None
        hidden_states_252 = query_states_36.to(torch.float32)
        query_states_36 = None
        pow_73 = hidden_states_252.pow(2)
        variance_72 = pow_73.mean(-1, keepdim=True)
        pow_73 = None
        add_144 = variance_72 + 1e-05
        variance_72 = None
        rsqrt_72 = torch.rsqrt(add_144)
        add_144 = None
        hidden_states_253 = hidden_states_252 * rsqrt_72
        hidden_states_252 = rsqrt_72 = None
        to_149 = hidden_states_253.to(torch.float16)
        hidden_states_253 = None
        query_states_37 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_
            * to_149
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_149
        ) = None
        hidden_states_254 = key_states_36.to(torch.float32)
        key_states_36 = None
        pow_74 = hidden_states_254.pow(2)
        variance_73 = pow_74.mean(-1, keepdim=True)
        pow_74 = None
        add_145 = variance_73 + 1e-05
        variance_73 = None
        rsqrt_73 = torch.rsqrt(add_145)
        add_145 = None
        hidden_states_255 = hidden_states_254 * rsqrt_73
        hidden_states_254 = rsqrt_73 = None
        to_151 = hidden_states_255.to(torch.float16)
        hidden_states_255 = None
        key_states_37 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_
            * to_151
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_151
        ) = None
        cos_21 = cos_2.unsqueeze(1)
        sin_21 = sin_2.unsqueeze(1)
        mul_240 = query_states_37 * cos_21
        x1_36 = query_states_37[(Ellipsis, slice(None, 32, None))]
        x2_36 = query_states_37[(Ellipsis, slice(32, None, None))]
        query_states_37 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_37 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_241 = cat_37 * sin_21
        cat_37 = None
        q_embed_18 = mul_240 + mul_241
        mul_240 = mul_241 = None
        mul_242 = key_states_37 * cos_21
        cos_21 = None
        x1_37 = key_states_37[(Ellipsis, slice(None, 32, None))]
        x2_37 = key_states_37[(Ellipsis, slice(32, None, None))]
        key_states_37 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_38 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_243 = cat_38 * sin_21
        cat_38 = sin_21 = None
        k_embed_18 = mul_242 + mul_243
        mul_242 = mul_243 = None
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
        hidden_states_256 = getitem_132.expand(1, 8, 4, 19, 64)
        getitem_132 = None
        key_36 = hidden_states_256.reshape(1, 32, 19, 64)
        hidden_states_256 = None
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
        hidden_states_257 = getitem_133.expand(1, 8, 4, 19, 64)
        getitem_133 = None
        value_36 = hidden_states_257.reshape(1, 32, 19, 64)
        hidden_states_257 = None
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
        item_36 = (
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_scaling = None
        attn_output_72 = torch._C._nn.scaled_dot_product_attention(
            query_18,
            key_37,
            value_37,
            attn_mask=attention_mask_19,
            dropout_p=0.0,
            scale=item_36,
            is_causal=False,
        )
        query_18 = key_37 = value_37 = attention_mask_19 = item_36 = None
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
        hidden_states_258 = attn_output_75.to(torch.float32)
        attn_output_75 = None
        pow_75 = hidden_states_258.pow(2)
        variance_74 = pow_75.mean(-1, keepdim=True)
        pow_75 = None
        item_37 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_148 = variance_74 + item_37
        variance_74 = item_37 = None
        rsqrt_74 = torch.rsqrt(add_148)
        add_148 = None
        hidden_states_259 = hidden_states_258 * rsqrt_74
        hidden_states_258 = rsqrt_74 = None
        to_153 = hidden_states_259.to(torch.float16)
        hidden_states_259 = None
        hidden_states_260 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
            * to_153
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = (
            to_153
        ) = None
        hidden_states_261 = hidden_states_251 + hidden_states_260
        hidden_states_251 = hidden_states_260 = None
        linear_130 = torch._C._nn.linear(
            hidden_states_261,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_130, inplace=False)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_261,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_246 = silu_18 * linear_131
        silu_18 = linear_131 = None
        down_proj_18 = torch._C._nn.linear(
            mul_246,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_246 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_262 = down_proj_18.to(torch.float32)
        down_proj_18 = None
        pow_76 = hidden_states_262.pow(2)
        variance_75 = pow_76.mean(-1, keepdim=True)
        pow_76 = None
        add_150 = variance_75 + 1e-05
        variance_75 = None
        rsqrt_75 = torch.rsqrt(add_150)
        add_150 = None
        hidden_states_263 = hidden_states_262 * rsqrt_75
        hidden_states_262 = rsqrt_75 = None
        to_155 = hidden_states_263.to(torch.float16)
        hidden_states_263 = None
        hidden_states_264 = (
            l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_
            * to_155
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_155
        ) = None
        hidden_states_265 = hidden_states_261 + hidden_states_264
        hidden_states_261 = hidden_states_264 = None
        linear_133 = torch._C._nn.linear(
            hidden_states_265,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_57 = linear_133.view((1, 19, -1, 64))
        linear_133 = None
        query_states_38 = view_57.transpose(1, 2)
        view_57 = None
        linear_134 = torch._C._nn.linear(
            hidden_states_265,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_58 = linear_134.view((1, 19, -1, 64))
        linear_134 = None
        key_states_38 = view_58.transpose(1, 2)
        view_58 = None
        linear_135 = torch._C._nn.linear(
            hidden_states_265,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_59 = linear_135.view((1, 19, -1, 64))
        linear_135 = None
        value_states_19 = view_59.transpose(1, 2)
        view_59 = None
        hidden_states_266 = query_states_38.to(torch.float32)
        query_states_38 = None
        pow_77 = hidden_states_266.pow(2)
        variance_76 = pow_77.mean(-1, keepdim=True)
        pow_77 = None
        add_152 = variance_76 + 1e-05
        variance_76 = None
        rsqrt_76 = torch.rsqrt(add_152)
        add_152 = None
        hidden_states_267 = hidden_states_266 * rsqrt_76
        hidden_states_266 = rsqrt_76 = None
        to_157 = hidden_states_267.to(torch.float16)
        hidden_states_267 = None
        query_states_39 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_
            * to_157
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_157
        ) = None
        hidden_states_268 = key_states_38.to(torch.float32)
        key_states_38 = None
        pow_78 = hidden_states_268.pow(2)
        variance_77 = pow_78.mean(-1, keepdim=True)
        pow_78 = None
        add_153 = variance_77 + 1e-05
        variance_77 = None
        rsqrt_77 = torch.rsqrt(add_153)
        add_153 = None
        hidden_states_269 = hidden_states_268 * rsqrt_77
        hidden_states_268 = rsqrt_77 = None
        to_159 = hidden_states_269.to(torch.float16)
        hidden_states_269 = None
        key_states_39 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_
            * to_159
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_159
        ) = None
        cos_22 = cos_2.unsqueeze(1)
        sin_22 = sin_2.unsqueeze(1)
        mul_253 = query_states_39 * cos_22
        x1_38 = query_states_39[(Ellipsis, slice(None, 32, None))]
        x2_38 = query_states_39[(Ellipsis, slice(32, None, None))]
        query_states_39 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_39 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_254 = cat_39 * sin_22
        cat_39 = None
        q_embed_19 = mul_253 + mul_254
        mul_253 = mul_254 = None
        mul_255 = key_states_39 * cos_22
        cos_22 = None
        x1_39 = key_states_39[(Ellipsis, slice(None, 32, None))]
        x2_39 = key_states_39[(Ellipsis, slice(32, None, None))]
        key_states_39 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_40 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_256 = cat_40 * sin_22
        cat_40 = sin_22 = None
        k_embed_19 = mul_255 + mul_256
        mul_255 = mul_256 = None
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
        hidden_states_270 = getitem_139.expand(1, 8, 4, 19, 64)
        getitem_139 = None
        key_38 = hidden_states_270.reshape(1, 32, 19, 64)
        hidden_states_270 = None
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
        hidden_states_271 = getitem_140.expand(1, 8, 4, 19, 64)
        getitem_140 = None
        value_38 = hidden_states_271.reshape(1, 32, 19, 64)
        hidden_states_271 = None
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
        item_38 = (
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_scaling = None
        attn_output_76 = torch._C._nn.scaled_dot_product_attention(
            query_19,
            key_39,
            value_39,
            attn_mask=attention_mask_20,
            dropout_p=0.0,
            scale=item_38,
            is_causal=False,
        )
        query_19 = key_39 = value_39 = attention_mask_20 = item_38 = None
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
        hidden_states_272 = attn_output_79.to(torch.float32)
        attn_output_79 = None
        pow_79 = hidden_states_272.pow(2)
        variance_78 = pow_79.mean(-1, keepdim=True)
        pow_79 = None
        item_39 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_156 = variance_78 + item_39
        variance_78 = item_39 = None
        rsqrt_78 = torch.rsqrt(add_156)
        add_156 = None
        hidden_states_273 = hidden_states_272 * rsqrt_78
        hidden_states_272 = rsqrt_78 = None
        to_161 = hidden_states_273.to(torch.float16)
        hidden_states_273 = None
        hidden_states_274 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
            * to_161
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = (
            to_161
        ) = None
        hidden_states_275 = hidden_states_265 + hidden_states_274
        hidden_states_265 = hidden_states_274 = None
        linear_137 = torch._C._nn.linear(
            hidden_states_275,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_137, inplace=False)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_275,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_259 = silu_19 * linear_138
        silu_19 = linear_138 = None
        down_proj_19 = torch._C._nn.linear(
            mul_259,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_259 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_276 = down_proj_19.to(torch.float32)
        down_proj_19 = None
        pow_80 = hidden_states_276.pow(2)
        variance_79 = pow_80.mean(-1, keepdim=True)
        pow_80 = None
        add_158 = variance_79 + 1e-05
        variance_79 = None
        rsqrt_79 = torch.rsqrt(add_158)
        add_158 = None
        hidden_states_277 = hidden_states_276 * rsqrt_79
        hidden_states_276 = rsqrt_79 = None
        to_163 = hidden_states_277.to(torch.float16)
        hidden_states_277 = None
        hidden_states_278 = (
            l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_
            * to_163
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_163
        ) = None
        hidden_states_279 = hidden_states_275 + hidden_states_278
        hidden_states_275 = hidden_states_278 = None
        linear_140 = torch._C._nn.linear(
            hidden_states_279,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_60 = linear_140.view((1, 19, -1, 64))
        linear_140 = None
        query_states_40 = view_60.transpose(1, 2)
        view_60 = None
        linear_141 = torch._C._nn.linear(
            hidden_states_279,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_61 = linear_141.view((1, 19, -1, 64))
        linear_141 = None
        key_states_40 = view_61.transpose(1, 2)
        view_61 = None
        linear_142 = torch._C._nn.linear(
            hidden_states_279,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_62 = linear_142.view((1, 19, -1, 64))
        linear_142 = None
        value_states_20 = view_62.transpose(1, 2)
        view_62 = None
        hidden_states_280 = query_states_40.to(torch.float32)
        query_states_40 = None
        pow_81 = hidden_states_280.pow(2)
        variance_80 = pow_81.mean(-1, keepdim=True)
        pow_81 = None
        add_160 = variance_80 + 1e-05
        variance_80 = None
        rsqrt_80 = torch.rsqrt(add_160)
        add_160 = None
        hidden_states_281 = hidden_states_280 * rsqrt_80
        hidden_states_280 = rsqrt_80 = None
        to_165 = hidden_states_281.to(torch.float16)
        hidden_states_281 = None
        query_states_41 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_
            * to_165
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_165
        ) = None
        hidden_states_282 = key_states_40.to(torch.float32)
        key_states_40 = None
        pow_82 = hidden_states_282.pow(2)
        variance_81 = pow_82.mean(-1, keepdim=True)
        pow_82 = None
        add_161 = variance_81 + 1e-05
        variance_81 = None
        rsqrt_81 = torch.rsqrt(add_161)
        add_161 = None
        hidden_states_283 = hidden_states_282 * rsqrt_81
        hidden_states_282 = rsqrt_81 = None
        to_167 = hidden_states_283.to(torch.float16)
        hidden_states_283 = None
        key_states_41 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_
            * to_167
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_167
        ) = None
        cos_23 = cos_2.unsqueeze(1)
        sin_23 = sin_2.unsqueeze(1)
        mul_266 = query_states_41 * cos_23
        x1_40 = query_states_41[(Ellipsis, slice(None, 32, None))]
        x2_40 = query_states_41[(Ellipsis, slice(32, None, None))]
        query_states_41 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_41 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_267 = cat_41 * sin_23
        cat_41 = None
        q_embed_20 = mul_266 + mul_267
        mul_266 = mul_267 = None
        mul_268 = key_states_41 * cos_23
        cos_23 = None
        x1_41 = key_states_41[(Ellipsis, slice(None, 32, None))]
        x2_41 = key_states_41[(Ellipsis, slice(32, None, None))]
        key_states_41 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_42 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_269 = cat_42 * sin_23
        cat_42 = sin_23 = None
        k_embed_20 = mul_268 + mul_269
        mul_268 = mul_269 = None
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
        hidden_states_284 = getitem_146.expand(1, 8, 4, 19, 64)
        getitem_146 = None
        key_40 = hidden_states_284.reshape(1, 32, 19, 64)
        hidden_states_284 = None
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
        hidden_states_285 = getitem_147.expand(1, 8, 4, 19, 64)
        getitem_147 = None
        value_40 = hidden_states_285.reshape(1, 32, 19, 64)
        hidden_states_285 = None
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
        item_40 = (
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_scaling = None
        attn_output_80 = torch._C._nn.scaled_dot_product_attention(
            query_20,
            key_41,
            value_41,
            attn_mask=attention_mask_21,
            dropout_p=0.0,
            scale=item_40,
            is_causal=False,
        )
        query_20 = key_41 = value_41 = attention_mask_21 = item_40 = None
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
        hidden_states_286 = attn_output_83.to(torch.float32)
        attn_output_83 = None
        pow_83 = hidden_states_286.pow(2)
        variance_82 = pow_83.mean(-1, keepdim=True)
        pow_83 = None
        item_41 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_164 = variance_82 + item_41
        variance_82 = item_41 = None
        rsqrt_82 = torch.rsqrt(add_164)
        add_164 = None
        hidden_states_287 = hidden_states_286 * rsqrt_82
        hidden_states_286 = rsqrt_82 = None
        to_169 = hidden_states_287.to(torch.float16)
        hidden_states_287 = None
        hidden_states_288 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
            * to_169
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = (
            to_169
        ) = None
        hidden_states_289 = hidden_states_279 + hidden_states_288
        hidden_states_279 = hidden_states_288 = None
        linear_144 = torch._C._nn.linear(
            hidden_states_289,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_144, inplace=False)
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_289,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_272 = silu_20 * linear_145
        silu_20 = linear_145 = None
        down_proj_20 = torch._C._nn.linear(
            mul_272,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_272 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_290 = down_proj_20.to(torch.float32)
        down_proj_20 = None
        pow_84 = hidden_states_290.pow(2)
        variance_83 = pow_84.mean(-1, keepdim=True)
        pow_84 = None
        add_166 = variance_83 + 1e-05
        variance_83 = None
        rsqrt_83 = torch.rsqrt(add_166)
        add_166 = None
        hidden_states_291 = hidden_states_290 * rsqrt_83
        hidden_states_290 = rsqrt_83 = None
        to_171 = hidden_states_291.to(torch.float16)
        hidden_states_291 = None
        hidden_states_292 = (
            l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_
            * to_171
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_171
        ) = None
        hidden_states_293 = hidden_states_289 + hidden_states_292
        hidden_states_289 = hidden_states_292 = None
        linear_147 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_63 = linear_147.view((1, 19, -1, 64))
        linear_147 = None
        query_states_42 = view_63.transpose(1, 2)
        view_63 = None
        linear_148 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_64 = linear_148.view((1, 19, -1, 64))
        linear_148 = None
        key_states_42 = view_64.transpose(1, 2)
        view_64 = None
        linear_149 = torch._C._nn.linear(
            hidden_states_293,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_65 = linear_149.view((1, 19, -1, 64))
        linear_149 = None
        value_states_21 = view_65.transpose(1, 2)
        view_65 = None
        hidden_states_294 = query_states_42.to(torch.float32)
        query_states_42 = None
        pow_85 = hidden_states_294.pow(2)
        variance_84 = pow_85.mean(-1, keepdim=True)
        pow_85 = None
        add_168 = variance_84 + 1e-05
        variance_84 = None
        rsqrt_84 = torch.rsqrt(add_168)
        add_168 = None
        hidden_states_295 = hidden_states_294 * rsqrt_84
        hidden_states_294 = rsqrt_84 = None
        to_173 = hidden_states_295.to(torch.float16)
        hidden_states_295 = None
        query_states_43 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_
            * to_173
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_173
        ) = None
        hidden_states_296 = key_states_42.to(torch.float32)
        key_states_42 = None
        pow_86 = hidden_states_296.pow(2)
        variance_85 = pow_86.mean(-1, keepdim=True)
        pow_86 = None
        add_169 = variance_85 + 1e-05
        variance_85 = None
        rsqrt_85 = torch.rsqrt(add_169)
        add_169 = None
        hidden_states_297 = hidden_states_296 * rsqrt_85
        hidden_states_296 = rsqrt_85 = None
        to_175 = hidden_states_297.to(torch.float16)
        hidden_states_297 = None
        key_states_43 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_
            * to_175
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_175
        ) = None
        cos_24 = cos_2.unsqueeze(1)
        sin_24 = sin_2.unsqueeze(1)
        mul_279 = query_states_43 * cos_24
        x1_42 = query_states_43[(Ellipsis, slice(None, 32, None))]
        x2_42 = query_states_43[(Ellipsis, slice(32, None, None))]
        query_states_43 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_43 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_280 = cat_43 * sin_24
        cat_43 = None
        q_embed_21 = mul_279 + mul_280
        mul_279 = mul_280 = None
        mul_281 = key_states_43 * cos_24
        cos_24 = None
        x1_43 = key_states_43[(Ellipsis, slice(None, 32, None))]
        x2_43 = key_states_43[(Ellipsis, slice(32, None, None))]
        key_states_43 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_44 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_282 = cat_44 * sin_24
        cat_44 = sin_24 = None
        k_embed_21 = mul_281 + mul_282
        mul_281 = mul_282 = None
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
        hidden_states_298 = getitem_153.expand(1, 8, 4, 19, 64)
        getitem_153 = None
        key_42 = hidden_states_298.reshape(1, 32, 19, 64)
        hidden_states_298 = None
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
        hidden_states_299 = getitem_154.expand(1, 8, 4, 19, 64)
        getitem_154 = None
        value_42 = hidden_states_299.reshape(1, 32, 19, 64)
        hidden_states_299 = None
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
        item_42 = (
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_scaling = None
        attn_output_84 = torch._C._nn.scaled_dot_product_attention(
            query_21,
            key_43,
            value_43,
            attn_mask=attention_mask_22,
            dropout_p=0.0,
            scale=item_42,
            is_causal=False,
        )
        query_21 = key_43 = value_43 = attention_mask_22 = item_42 = None
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
        hidden_states_300 = attn_output_87.to(torch.float32)
        attn_output_87 = None
        pow_87 = hidden_states_300.pow(2)
        variance_86 = pow_87.mean(-1, keepdim=True)
        pow_87 = None
        item_43 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_172 = variance_86 + item_43
        variance_86 = item_43 = None
        rsqrt_86 = torch.rsqrt(add_172)
        add_172 = None
        hidden_states_301 = hidden_states_300 * rsqrt_86
        hidden_states_300 = rsqrt_86 = None
        to_177 = hidden_states_301.to(torch.float16)
        hidden_states_301 = None
        hidden_states_302 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
            * to_177
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = (
            to_177
        ) = None
        hidden_states_303 = hidden_states_293 + hidden_states_302
        hidden_states_293 = hidden_states_302 = None
        linear_151 = torch._C._nn.linear(
            hidden_states_303,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_151, inplace=False)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_303,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_285 = silu_21 * linear_152
        silu_21 = linear_152 = None
        down_proj_21 = torch._C._nn.linear(
            mul_285,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_285 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_304 = down_proj_21.to(torch.float32)
        down_proj_21 = None
        pow_88 = hidden_states_304.pow(2)
        variance_87 = pow_88.mean(-1, keepdim=True)
        pow_88 = None
        add_174 = variance_87 + 1e-05
        variance_87 = None
        rsqrt_87 = torch.rsqrt(add_174)
        add_174 = None
        hidden_states_305 = hidden_states_304 * rsqrt_87
        hidden_states_304 = rsqrt_87 = None
        to_179 = hidden_states_305.to(torch.float16)
        hidden_states_305 = None
        hidden_states_306 = (
            l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_
            * to_179
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_179
        ) = None
        hidden_states_307 = hidden_states_303 + hidden_states_306
        hidden_states_303 = hidden_states_306 = None
        linear_154 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_66 = linear_154.view((1, 19, -1, 64))
        linear_154 = None
        query_states_44 = view_66.transpose(1, 2)
        view_66 = None
        linear_155 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_67 = linear_155.view((1, 19, -1, 64))
        linear_155 = None
        key_states_44 = view_67.transpose(1, 2)
        view_67 = None
        linear_156 = torch._C._nn.linear(
            hidden_states_307,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_68 = linear_156.view((1, 19, -1, 64))
        linear_156 = None
        value_states_22 = view_68.transpose(1, 2)
        view_68 = None
        hidden_states_308 = query_states_44.to(torch.float32)
        query_states_44 = None
        pow_89 = hidden_states_308.pow(2)
        variance_88 = pow_89.mean(-1, keepdim=True)
        pow_89 = None
        add_176 = variance_88 + 1e-05
        variance_88 = None
        rsqrt_88 = torch.rsqrt(add_176)
        add_176 = None
        hidden_states_309 = hidden_states_308 * rsqrt_88
        hidden_states_308 = rsqrt_88 = None
        to_181 = hidden_states_309.to(torch.float16)
        hidden_states_309 = None
        query_states_45 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_
            * to_181
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_181
        ) = None
        hidden_states_310 = key_states_44.to(torch.float32)
        key_states_44 = None
        pow_90 = hidden_states_310.pow(2)
        variance_89 = pow_90.mean(-1, keepdim=True)
        pow_90 = None
        add_177 = variance_89 + 1e-05
        variance_89 = None
        rsqrt_89 = torch.rsqrt(add_177)
        add_177 = None
        hidden_states_311 = hidden_states_310 * rsqrt_89
        hidden_states_310 = rsqrt_89 = None
        to_183 = hidden_states_311.to(torch.float16)
        hidden_states_311 = None
        key_states_45 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_
            * to_183
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_183
        ) = None
        cos_25 = cos_2.unsqueeze(1)
        sin_25 = sin_2.unsqueeze(1)
        mul_292 = query_states_45 * cos_25
        x1_44 = query_states_45[(Ellipsis, slice(None, 32, None))]
        x2_44 = query_states_45[(Ellipsis, slice(32, None, None))]
        query_states_45 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_45 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_293 = cat_45 * sin_25
        cat_45 = None
        q_embed_22 = mul_292 + mul_293
        mul_292 = mul_293 = None
        mul_294 = key_states_45 * cos_25
        cos_25 = None
        x1_45 = key_states_45[(Ellipsis, slice(None, 32, None))]
        x2_45 = key_states_45[(Ellipsis, slice(32, None, None))]
        key_states_45 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_46 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_295 = cat_46 * sin_25
        cat_46 = sin_25 = None
        k_embed_22 = mul_294 + mul_295
        mul_294 = mul_295 = None
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
        hidden_states_312 = getitem_160.expand(1, 8, 4, 19, 64)
        getitem_160 = None
        key_44 = hidden_states_312.reshape(1, 32, 19, 64)
        hidden_states_312 = None
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
        hidden_states_313 = getitem_161.expand(1, 8, 4, 19, 64)
        getitem_161 = None
        value_44 = hidden_states_313.reshape(1, 32, 19, 64)
        hidden_states_313 = None
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
        item_44 = (
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_scaling = None
        attn_output_88 = torch._C._nn.scaled_dot_product_attention(
            query_22,
            key_45,
            value_45,
            attn_mask=attention_mask_23,
            dropout_p=0.0,
            scale=item_44,
            is_causal=False,
        )
        query_22 = key_45 = value_45 = attention_mask_23 = item_44 = None
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
        hidden_states_314 = attn_output_91.to(torch.float32)
        attn_output_91 = None
        pow_91 = hidden_states_314.pow(2)
        variance_90 = pow_91.mean(-1, keepdim=True)
        pow_91 = None
        item_45 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_180 = variance_90 + item_45
        variance_90 = item_45 = None
        rsqrt_90 = torch.rsqrt(add_180)
        add_180 = None
        hidden_states_315 = hidden_states_314 * rsqrt_90
        hidden_states_314 = rsqrt_90 = None
        to_185 = hidden_states_315.to(torch.float16)
        hidden_states_315 = None
        hidden_states_316 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
            * to_185
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = (
            to_185
        ) = None
        hidden_states_317 = hidden_states_307 + hidden_states_316
        hidden_states_307 = hidden_states_316 = None
        linear_158 = torch._C._nn.linear(
            hidden_states_317,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_158, inplace=False)
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_317,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_298 = silu_22 * linear_159
        silu_22 = linear_159 = None
        down_proj_22 = torch._C._nn.linear(
            mul_298,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_298 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_318 = down_proj_22.to(torch.float32)
        down_proj_22 = None
        pow_92 = hidden_states_318.pow(2)
        variance_91 = pow_92.mean(-1, keepdim=True)
        pow_92 = None
        add_182 = variance_91 + 1e-05
        variance_91 = None
        rsqrt_91 = torch.rsqrt(add_182)
        add_182 = None
        hidden_states_319 = hidden_states_318 * rsqrt_91
        hidden_states_318 = rsqrt_91 = None
        to_187 = hidden_states_319.to(torch.float16)
        hidden_states_319 = None
        hidden_states_320 = (
            l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_
            * to_187
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_187
        ) = None
        hidden_states_321 = hidden_states_317 + hidden_states_320
        hidden_states_317 = hidden_states_320 = None
        linear_161 = torch._C._nn.linear(
            hidden_states_321,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_69 = linear_161.view((1, 19, -1, 64))
        linear_161 = None
        query_states_46 = view_69.transpose(1, 2)
        view_69 = None
        linear_162 = torch._C._nn.linear(
            hidden_states_321,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_70 = linear_162.view((1, 19, -1, 64))
        linear_162 = None
        key_states_46 = view_70.transpose(1, 2)
        view_70 = None
        linear_163 = torch._C._nn.linear(
            hidden_states_321,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_71 = linear_163.view((1, 19, -1, 64))
        linear_163 = None
        value_states_23 = view_71.transpose(1, 2)
        view_71 = None
        hidden_states_322 = query_states_46.to(torch.float32)
        query_states_46 = None
        pow_93 = hidden_states_322.pow(2)
        variance_92 = pow_93.mean(-1, keepdim=True)
        pow_93 = None
        add_184 = variance_92 + 1e-05
        variance_92 = None
        rsqrt_92 = torch.rsqrt(add_184)
        add_184 = None
        hidden_states_323 = hidden_states_322 * rsqrt_92
        hidden_states_322 = rsqrt_92 = None
        to_189 = hidden_states_323.to(torch.float16)
        hidden_states_323 = None
        query_states_47 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_
            * to_189
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_189
        ) = None
        hidden_states_324 = key_states_46.to(torch.float32)
        key_states_46 = None
        pow_94 = hidden_states_324.pow(2)
        variance_93 = pow_94.mean(-1, keepdim=True)
        pow_94 = None
        add_185 = variance_93 + 1e-05
        variance_93 = None
        rsqrt_93 = torch.rsqrt(add_185)
        add_185 = None
        hidden_states_325 = hidden_states_324 * rsqrt_93
        hidden_states_324 = rsqrt_93 = None
        to_191 = hidden_states_325.to(torch.float16)
        hidden_states_325 = None
        key_states_47 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_
            * to_191
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_191
        ) = None
        cos_26 = cos_2.unsqueeze(1)
        sin_26 = sin_2.unsqueeze(1)
        mul_305 = query_states_47 * cos_26
        x1_46 = query_states_47[(Ellipsis, slice(None, 32, None))]
        x2_46 = query_states_47[(Ellipsis, slice(32, None, None))]
        query_states_47 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_47 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_306 = cat_47 * sin_26
        cat_47 = None
        q_embed_23 = mul_305 + mul_306
        mul_305 = mul_306 = None
        mul_307 = key_states_47 * cos_26
        cos_26 = None
        x1_47 = key_states_47[(Ellipsis, slice(None, 32, None))]
        x2_47 = key_states_47[(Ellipsis, slice(32, None, None))]
        key_states_47 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_48 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_308 = cat_48 * sin_26
        cat_48 = sin_26 = None
        k_embed_23 = mul_307 + mul_308
        mul_307 = mul_308 = None
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
        hidden_states_326 = getitem_167.expand(1, 8, 4, 19, 64)
        getitem_167 = None
        key_46 = hidden_states_326.reshape(1, 32, 19, 64)
        hidden_states_326 = None
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
        hidden_states_327 = getitem_168.expand(1, 8, 4, 19, 64)
        getitem_168 = None
        value_46 = hidden_states_327.reshape(1, 32, 19, 64)
        hidden_states_327 = None
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
        item_46 = (
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_scaling = None
        attn_output_92 = torch._C._nn.scaled_dot_product_attention(
            query_23,
            key_47,
            value_47,
            attn_mask=attention_mask_24,
            dropout_p=0.0,
            scale=item_46,
            is_causal=False,
        )
        query_23 = key_47 = value_47 = attention_mask_24 = item_46 = None
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
        hidden_states_328 = attn_output_95.to(torch.float32)
        attn_output_95 = None
        pow_95 = hidden_states_328.pow(2)
        variance_94 = pow_95.mean(-1, keepdim=True)
        pow_95 = None
        item_47 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_188 = variance_94 + item_47
        variance_94 = item_47 = None
        rsqrt_94 = torch.rsqrt(add_188)
        add_188 = None
        hidden_states_329 = hidden_states_328 * rsqrt_94
        hidden_states_328 = rsqrt_94 = None
        to_193 = hidden_states_329.to(torch.float16)
        hidden_states_329 = None
        hidden_states_330 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
            * to_193
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = (
            to_193
        ) = None
        hidden_states_331 = hidden_states_321 + hidden_states_330
        hidden_states_321 = hidden_states_330 = None
        linear_165 = torch._C._nn.linear(
            hidden_states_331,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_165, inplace=False)
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_331,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_311 = silu_23 * linear_166
        silu_23 = linear_166 = None
        down_proj_23 = torch._C._nn.linear(
            mul_311,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_311 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_332 = down_proj_23.to(torch.float32)
        down_proj_23 = None
        pow_96 = hidden_states_332.pow(2)
        variance_95 = pow_96.mean(-1, keepdim=True)
        pow_96 = None
        add_190 = variance_95 + 1e-05
        variance_95 = None
        rsqrt_95 = torch.rsqrt(add_190)
        add_190 = None
        hidden_states_333 = hidden_states_332 * rsqrt_95
        hidden_states_332 = rsqrt_95 = None
        to_195 = hidden_states_333.to(torch.float16)
        hidden_states_333 = None
        hidden_states_334 = (
            l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_
            * to_195
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_195
        ) = None
        hidden_states_335 = hidden_states_331 + hidden_states_334
        hidden_states_331 = hidden_states_334 = None
        linear_168 = torch._C._nn.linear(
            hidden_states_335,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_72 = linear_168.view((1, 19, -1, 64))
        linear_168 = None
        query_states_48 = view_72.transpose(1, 2)
        view_72 = None
        linear_169 = torch._C._nn.linear(
            hidden_states_335,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_73 = linear_169.view((1, 19, -1, 64))
        linear_169 = None
        key_states_48 = view_73.transpose(1, 2)
        view_73 = None
        linear_170 = torch._C._nn.linear(
            hidden_states_335,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_74 = linear_170.view((1, 19, -1, 64))
        linear_170 = None
        value_states_24 = view_74.transpose(1, 2)
        view_74 = None
        hidden_states_336 = query_states_48.to(torch.float32)
        query_states_48 = None
        pow_97 = hidden_states_336.pow(2)
        variance_96 = pow_97.mean(-1, keepdim=True)
        pow_97 = None
        add_192 = variance_96 + 1e-05
        variance_96 = None
        rsqrt_96 = torch.rsqrt(add_192)
        add_192 = None
        hidden_states_337 = hidden_states_336 * rsqrt_96
        hidden_states_336 = rsqrt_96 = None
        to_197 = hidden_states_337.to(torch.float16)
        hidden_states_337 = None
        query_states_49 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_
            * to_197
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_197
        ) = None
        hidden_states_338 = key_states_48.to(torch.float32)
        key_states_48 = None
        pow_98 = hidden_states_338.pow(2)
        variance_97 = pow_98.mean(-1, keepdim=True)
        pow_98 = None
        add_193 = variance_97 + 1e-05
        variance_97 = None
        rsqrt_97 = torch.rsqrt(add_193)
        add_193 = None
        hidden_states_339 = hidden_states_338 * rsqrt_97
        hidden_states_338 = rsqrt_97 = None
        to_199 = hidden_states_339.to(torch.float16)
        hidden_states_339 = None
        key_states_49 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_
            * to_199
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_199
        ) = None
        cos_27 = cos_2.unsqueeze(1)
        sin_27 = sin_2.unsqueeze(1)
        mul_318 = query_states_49 * cos_27
        x1_48 = query_states_49[(Ellipsis, slice(None, 32, None))]
        x2_48 = query_states_49[(Ellipsis, slice(32, None, None))]
        query_states_49 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_49 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_319 = cat_49 * sin_27
        cat_49 = None
        q_embed_24 = mul_318 + mul_319
        mul_318 = mul_319 = None
        mul_320 = key_states_49 * cos_27
        cos_27 = None
        x1_49 = key_states_49[(Ellipsis, slice(None, 32, None))]
        x2_49 = key_states_49[(Ellipsis, slice(32, None, None))]
        key_states_49 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_50 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_321 = cat_50 * sin_27
        cat_50 = sin_27 = None
        k_embed_24 = mul_320 + mul_321
        mul_320 = mul_321 = None
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
        hidden_states_340 = getitem_174.expand(1, 8, 4, 19, 64)
        getitem_174 = None
        key_48 = hidden_states_340.reshape(1, 32, 19, 64)
        hidden_states_340 = None
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
        hidden_states_341 = getitem_175.expand(1, 8, 4, 19, 64)
        getitem_175 = None
        value_48 = hidden_states_341.reshape(1, 32, 19, 64)
        hidden_states_341 = None
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
        item_48 = (
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_scaling = None
        attn_output_96 = torch._C._nn.scaled_dot_product_attention(
            query_24,
            key_49,
            value_49,
            attn_mask=attention_mask_25,
            dropout_p=0.0,
            scale=item_48,
            is_causal=False,
        )
        query_24 = key_49 = value_49 = attention_mask_25 = item_48 = None
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
        hidden_states_342 = attn_output_99.to(torch.float32)
        attn_output_99 = None
        pow_99 = hidden_states_342.pow(2)
        variance_98 = pow_99.mean(-1, keepdim=True)
        pow_99 = None
        item_49 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_196 = variance_98 + item_49
        variance_98 = item_49 = None
        rsqrt_98 = torch.rsqrt(add_196)
        add_196 = None
        hidden_states_343 = hidden_states_342 * rsqrt_98
        hidden_states_342 = rsqrt_98 = None
        to_201 = hidden_states_343.to(torch.float16)
        hidden_states_343 = None
        hidden_states_344 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
            * to_201
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = (
            to_201
        ) = None
        hidden_states_345 = hidden_states_335 + hidden_states_344
        hidden_states_335 = hidden_states_344 = None
        linear_172 = torch._C._nn.linear(
            hidden_states_345,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_172, inplace=False)
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_345,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_324 = silu_24 * linear_173
        silu_24 = linear_173 = None
        down_proj_24 = torch._C._nn.linear(
            mul_324,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_324 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_346 = down_proj_24.to(torch.float32)
        down_proj_24 = None
        pow_100 = hidden_states_346.pow(2)
        variance_99 = pow_100.mean(-1, keepdim=True)
        pow_100 = None
        add_198 = variance_99 + 1e-05
        variance_99 = None
        rsqrt_99 = torch.rsqrt(add_198)
        add_198 = None
        hidden_states_347 = hidden_states_346 * rsqrt_99
        hidden_states_346 = rsqrt_99 = None
        to_203 = hidden_states_347.to(torch.float16)
        hidden_states_347 = None
        hidden_states_348 = (
            l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_
            * to_203
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_203
        ) = None
        hidden_states_349 = hidden_states_345 + hidden_states_348
        hidden_states_345 = hidden_states_348 = None
        linear_175 = torch._C._nn.linear(
            hidden_states_349,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_75 = linear_175.view((1, 19, -1, 64))
        linear_175 = None
        query_states_50 = view_75.transpose(1, 2)
        view_75 = None
        linear_176 = torch._C._nn.linear(
            hidden_states_349,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_76 = linear_176.view((1, 19, -1, 64))
        linear_176 = None
        key_states_50 = view_76.transpose(1, 2)
        view_76 = None
        linear_177 = torch._C._nn.linear(
            hidden_states_349,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_77 = linear_177.view((1, 19, -1, 64))
        linear_177 = None
        value_states_25 = view_77.transpose(1, 2)
        view_77 = None
        hidden_states_350 = query_states_50.to(torch.float32)
        query_states_50 = None
        pow_101 = hidden_states_350.pow(2)
        variance_100 = pow_101.mean(-1, keepdim=True)
        pow_101 = None
        add_200 = variance_100 + 1e-05
        variance_100 = None
        rsqrt_100 = torch.rsqrt(add_200)
        add_200 = None
        hidden_states_351 = hidden_states_350 * rsqrt_100
        hidden_states_350 = rsqrt_100 = None
        to_205 = hidden_states_351.to(torch.float16)
        hidden_states_351 = None
        query_states_51 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_
            * to_205
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_205
        ) = None
        hidden_states_352 = key_states_50.to(torch.float32)
        key_states_50 = None
        pow_102 = hidden_states_352.pow(2)
        variance_101 = pow_102.mean(-1, keepdim=True)
        pow_102 = None
        add_201 = variance_101 + 1e-05
        variance_101 = None
        rsqrt_101 = torch.rsqrt(add_201)
        add_201 = None
        hidden_states_353 = hidden_states_352 * rsqrt_101
        hidden_states_352 = rsqrt_101 = None
        to_207 = hidden_states_353.to(torch.float16)
        hidden_states_353 = None
        key_states_51 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_
            * to_207
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_207
        ) = None
        cos_28 = cos_2.unsqueeze(1)
        sin_28 = sin_2.unsqueeze(1)
        mul_331 = query_states_51 * cos_28
        x1_50 = query_states_51[(Ellipsis, slice(None, 32, None))]
        x2_50 = query_states_51[(Ellipsis, slice(32, None, None))]
        query_states_51 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_51 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_332 = cat_51 * sin_28
        cat_51 = None
        q_embed_25 = mul_331 + mul_332
        mul_331 = mul_332 = None
        mul_333 = key_states_51 * cos_28
        cos_28 = None
        x1_51 = key_states_51[(Ellipsis, slice(None, 32, None))]
        x2_51 = key_states_51[(Ellipsis, slice(32, None, None))]
        key_states_51 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_52 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_334 = cat_52 * sin_28
        cat_52 = sin_28 = None
        k_embed_25 = mul_333 + mul_334
        mul_333 = mul_334 = None
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
        hidden_states_354 = getitem_181.expand(1, 8, 4, 19, 64)
        getitem_181 = None
        key_50 = hidden_states_354.reshape(1, 32, 19, 64)
        hidden_states_354 = None
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
        hidden_states_355 = getitem_182.expand(1, 8, 4, 19, 64)
        getitem_182 = None
        value_50 = hidden_states_355.reshape(1, 32, 19, 64)
        hidden_states_355 = None
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
        item_50 = (
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_scaling = None
        attn_output_100 = torch._C._nn.scaled_dot_product_attention(
            query_25,
            key_51,
            value_51,
            attn_mask=attention_mask_26,
            dropout_p=0.0,
            scale=item_50,
            is_causal=False,
        )
        query_25 = key_51 = value_51 = attention_mask_26 = item_50 = None
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
        hidden_states_356 = attn_output_103.to(torch.float32)
        attn_output_103 = None
        pow_103 = hidden_states_356.pow(2)
        variance_102 = pow_103.mean(-1, keepdim=True)
        pow_103 = None
        item_51 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_204 = variance_102 + item_51
        variance_102 = item_51 = None
        rsqrt_102 = torch.rsqrt(add_204)
        add_204 = None
        hidden_states_357 = hidden_states_356 * rsqrt_102
        hidden_states_356 = rsqrt_102 = None
        to_209 = hidden_states_357.to(torch.float16)
        hidden_states_357 = None
        hidden_states_358 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
            * to_209
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = (
            to_209
        ) = None
        hidden_states_359 = hidden_states_349 + hidden_states_358
        hidden_states_349 = hidden_states_358 = None
        linear_179 = torch._C._nn.linear(
            hidden_states_359,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_179, inplace=False)
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_359,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_337 = silu_25 * linear_180
        silu_25 = linear_180 = None
        down_proj_25 = torch._C._nn.linear(
            mul_337,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_337 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_360 = down_proj_25.to(torch.float32)
        down_proj_25 = None
        pow_104 = hidden_states_360.pow(2)
        variance_103 = pow_104.mean(-1, keepdim=True)
        pow_104 = None
        add_206 = variance_103 + 1e-05
        variance_103 = None
        rsqrt_103 = torch.rsqrt(add_206)
        add_206 = None
        hidden_states_361 = hidden_states_360 * rsqrt_103
        hidden_states_360 = rsqrt_103 = None
        to_211 = hidden_states_361.to(torch.float16)
        hidden_states_361 = None
        hidden_states_362 = (
            l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_
            * to_211
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_211
        ) = None
        hidden_states_363 = hidden_states_359 + hidden_states_362
        hidden_states_359 = hidden_states_362 = None
        linear_182 = torch._C._nn.linear(
            hidden_states_363,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_78 = linear_182.view((1, 19, -1, 64))
        linear_182 = None
        query_states_52 = view_78.transpose(1, 2)
        view_78 = None
        linear_183 = torch._C._nn.linear(
            hidden_states_363,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_79 = linear_183.view((1, 19, -1, 64))
        linear_183 = None
        key_states_52 = view_79.transpose(1, 2)
        view_79 = None
        linear_184 = torch._C._nn.linear(
            hidden_states_363,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_80 = linear_184.view((1, 19, -1, 64))
        linear_184 = None
        value_states_26 = view_80.transpose(1, 2)
        view_80 = None
        hidden_states_364 = query_states_52.to(torch.float32)
        query_states_52 = None
        pow_105 = hidden_states_364.pow(2)
        variance_104 = pow_105.mean(-1, keepdim=True)
        pow_105 = None
        add_208 = variance_104 + 1e-05
        variance_104 = None
        rsqrt_104 = torch.rsqrt(add_208)
        add_208 = None
        hidden_states_365 = hidden_states_364 * rsqrt_104
        hidden_states_364 = rsqrt_104 = None
        to_213 = hidden_states_365.to(torch.float16)
        hidden_states_365 = None
        query_states_53 = (
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_
            * to_213
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_213
        ) = None
        hidden_states_366 = key_states_52.to(torch.float32)
        key_states_52 = None
        pow_106 = hidden_states_366.pow(2)
        variance_105 = pow_106.mean(-1, keepdim=True)
        pow_106 = None
        add_209 = variance_105 + 1e-05
        variance_105 = None
        rsqrt_105 = torch.rsqrt(add_209)
        add_209 = None
        hidden_states_367 = hidden_states_366 * rsqrt_105
        hidden_states_366 = rsqrt_105 = None
        to_215 = hidden_states_367.to(torch.float16)
        hidden_states_367 = None
        key_states_53 = (
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_
            * to_215
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_215
        ) = None
        cos_29 = cos_2.unsqueeze(1)
        sin_29 = sin_2.unsqueeze(1)
        mul_344 = query_states_53 * cos_29
        x1_52 = query_states_53[(Ellipsis, slice(None, 32, None))]
        x2_52 = query_states_53[(Ellipsis, slice(32, None, None))]
        query_states_53 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_53 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_345 = cat_53 * sin_29
        cat_53 = None
        q_embed_26 = mul_344 + mul_345
        mul_344 = mul_345 = None
        mul_346 = key_states_53 * cos_29
        cos_29 = None
        x1_53 = key_states_53[(Ellipsis, slice(None, 32, None))]
        x2_53 = key_states_53[(Ellipsis, slice(32, None, None))]
        key_states_53 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_54 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_347 = cat_54 * sin_29
        cat_54 = sin_29 = None
        k_embed_26 = mul_346 + mul_347
        mul_346 = mul_347 = None
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
        hidden_states_368 = getitem_188.expand(1, 8, 4, 19, 64)
        getitem_188 = None
        key_52 = hidden_states_368.reshape(1, 32, 19, 64)
        hidden_states_368 = None
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
        hidden_states_369 = getitem_189.expand(1, 8, 4, 19, 64)
        getitem_189 = None
        value_52 = hidden_states_369.reshape(1, 32, 19, 64)
        hidden_states_369 = None
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
        item_52 = (
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_scaling = None
        attn_output_104 = torch._C._nn.scaled_dot_product_attention(
            query_26,
            key_53,
            value_53,
            attn_mask=attention_mask_27,
            dropout_p=0.0,
            scale=item_52,
            is_causal=False,
        )
        query_26 = key_53 = value_53 = attention_mask_27 = item_52 = None
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
        hidden_states_370 = attn_output_107.to(torch.float32)
        attn_output_107 = None
        pow_107 = hidden_states_370.pow(2)
        variance_106 = pow_107.mean(-1, keepdim=True)
        pow_107 = None
        item_53 = (
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_212 = variance_106 + item_53
        variance_106 = item_53 = None
        rsqrt_106 = torch.rsqrt(add_212)
        add_212 = None
        hidden_states_371 = hidden_states_370 * rsqrt_106
        hidden_states_370 = rsqrt_106 = None
        to_217 = hidden_states_371.to(torch.float16)
        hidden_states_371 = None
        hidden_states_372 = (
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
            * to_217
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = (
            to_217
        ) = None
        hidden_states_373 = hidden_states_363 + hidden_states_372
        hidden_states_363 = hidden_states_372 = None
        linear_186 = torch._C._nn.linear(
            hidden_states_373,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_186, inplace=False)
        linear_186 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_373,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_350 = silu_26 * linear_187
        silu_26 = linear_187 = None
        down_proj_26 = torch._C._nn.linear(
            mul_350,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_350 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_374 = down_proj_26.to(torch.float32)
        down_proj_26 = None
        pow_108 = hidden_states_374.pow(2)
        variance_107 = pow_108.mean(-1, keepdim=True)
        pow_108 = None
        add_214 = variance_107 + 1e-05
        variance_107 = None
        rsqrt_107 = torch.rsqrt(add_214)
        add_214 = None
        hidden_states_375 = hidden_states_374 * rsqrt_107
        hidden_states_374 = rsqrt_107 = None
        to_219 = hidden_states_375.to(torch.float16)
        hidden_states_375 = None
        hidden_states_376 = (
            l_self_modules_model_modules_layers_modules_26_modules_post_feedforward_layernorm_parameters_weight_
            * to_219
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_219
        ) = None
        hidden_states_377 = hidden_states_373 + hidden_states_376
        hidden_states_373 = hidden_states_376 = None
        linear_189 = torch._C._nn.linear(
            hidden_states_377,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_81 = linear_189.view((1, 19, -1, 64))
        linear_189 = None
        query_states_54 = view_81.transpose(1, 2)
        view_81 = None
        linear_190 = torch._C._nn.linear(
            hidden_states_377,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_82 = linear_190.view((1, 19, -1, 64))
        linear_190 = None
        key_states_54 = view_82.transpose(1, 2)
        view_82 = None
        linear_191 = torch._C._nn.linear(
            hidden_states_377,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_83 = linear_191.view((1, 19, -1, 64))
        linear_191 = None
        value_states_27 = view_83.transpose(1, 2)
        view_83 = None
        hidden_states_378 = query_states_54.to(torch.float32)
        query_states_54 = None
        pow_109 = hidden_states_378.pow(2)
        variance_108 = pow_109.mean(-1, keepdim=True)
        pow_109 = None
        add_216 = variance_108 + 1e-05
        variance_108 = None
        rsqrt_108 = torch.rsqrt(add_216)
        add_216 = None
        hidden_states_379 = hidden_states_378 * rsqrt_108
        hidden_states_378 = rsqrt_108 = None
        to_221 = hidden_states_379.to(torch.float16)
        hidden_states_379 = None
        query_states_55 = (
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_
            * to_221
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_221
        ) = None
        hidden_states_380 = key_states_54.to(torch.float32)
        key_states_54 = None
        pow_110 = hidden_states_380.pow(2)
        variance_109 = pow_110.mean(-1, keepdim=True)
        pow_110 = None
        add_217 = variance_109 + 1e-05
        variance_109 = None
        rsqrt_109 = torch.rsqrt(add_217)
        add_217 = None
        hidden_states_381 = hidden_states_380 * rsqrt_109
        hidden_states_380 = rsqrt_109 = None
        to_223 = hidden_states_381.to(torch.float16)
        hidden_states_381 = None
        key_states_55 = (
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_
            * to_223
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_223
        ) = None
        cos_30 = cos_2.unsqueeze(1)
        sin_30 = sin_2.unsqueeze(1)
        mul_357 = query_states_55 * cos_30
        x1_54 = query_states_55[(Ellipsis, slice(None, 32, None))]
        x2_54 = query_states_55[(Ellipsis, slice(32, None, None))]
        query_states_55 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_55 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_358 = cat_55 * sin_30
        cat_55 = None
        q_embed_27 = mul_357 + mul_358
        mul_357 = mul_358 = None
        mul_359 = key_states_55 * cos_30
        cos_30 = None
        x1_55 = key_states_55[(Ellipsis, slice(None, 32, None))]
        x2_55 = key_states_55[(Ellipsis, slice(32, None, None))]
        key_states_55 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_56 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_360 = cat_56 * sin_30
        cat_56 = sin_30 = None
        k_embed_27 = mul_359 + mul_360
        mul_359 = mul_360 = None
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
        hidden_states_382 = getitem_195.expand(1, 8, 4, 19, 64)
        getitem_195 = None
        key_54 = hidden_states_382.reshape(1, 32, 19, 64)
        hidden_states_382 = None
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
        hidden_states_383 = getitem_196.expand(1, 8, 4, 19, 64)
        getitem_196 = None
        value_54 = hidden_states_383.reshape(1, 32, 19, 64)
        hidden_states_383 = None
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
        item_54 = (
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_scaling.item()
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_scaling = None
        attn_output_108 = torch._C._nn.scaled_dot_product_attention(
            query_27,
            key_55,
            value_55,
            attn_mask=attention_mask_28,
            dropout_p=0.0,
            scale=item_54,
            is_causal=False,
        )
        query_27 = key_55 = value_55 = attention_mask_28 = item_54 = None
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
        hidden_states_384 = attn_output_111.to(torch.float32)
        attn_output_111 = None
        pow_111 = hidden_states_384.pow(2)
        variance_110 = pow_111.mean(-1, keepdim=True)
        pow_111 = None
        item_55 = (
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_variance_epsilon.item()
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_variance_epsilon = (
            None
        )
        add_220 = variance_110 + item_55
        variance_110 = item_55 = None
        rsqrt_110 = torch.rsqrt(add_220)
        add_220 = None
        hidden_states_385 = hidden_states_384 * rsqrt_110
        hidden_states_384 = rsqrt_110 = None
        to_225 = hidden_states_385.to(torch.float16)
        hidden_states_385 = None
        hidden_states_386 = (
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
            * to_225
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = (
            to_225
        ) = None
        hidden_states_387 = hidden_states_377 + hidden_states_386
        hidden_states_377 = hidden_states_386 = None
        linear_193 = torch._C._nn.linear(
            hidden_states_387,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_193, inplace=False)
        linear_193 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_387,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_363 = silu_27 * linear_194
        silu_27 = linear_194 = None
        down_proj_27 = torch._C._nn.linear(
            mul_363,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_363 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_388 = down_proj_27.to(torch.float32)
        down_proj_27 = None
        pow_112 = hidden_states_388.pow(2)
        variance_111 = pow_112.mean(-1, keepdim=True)
        pow_112 = None
        add_222 = variance_111 + 1e-05
        variance_111 = None
        rsqrt_111 = torch.rsqrt(add_222)
        add_222 = None
        hidden_states_389 = hidden_states_388 * rsqrt_111
        hidden_states_388 = rsqrt_111 = None
        to_227 = hidden_states_389.to(torch.float16)
        hidden_states_389 = None
        hidden_states_390 = (
            l_self_modules_model_modules_layers_modules_27_modules_post_feedforward_layernorm_parameters_weight_
            * to_227
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_227
        ) = None
        hidden_states_391 = hidden_states_387 + hidden_states_390
        hidden_states_387 = hidden_states_390 = None
        linear_196 = torch._C._nn.linear(
            hidden_states_391,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_84 = linear_196.view((1, 19, -1, 64))
        linear_196 = None
        query_states_56 = view_84.transpose(1, 2)
        view_84 = None
        linear_197 = torch._C._nn.linear(
            hidden_states_391,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_85 = linear_197.view((1, 19, -1, 64))
        linear_197 = None
        key_states_56 = view_85.transpose(1, 2)
        view_85 = None
        linear_198 = torch._C._nn.linear(
            hidden_states_391,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_86 = linear_198.view((1, 19, -1, 64))
        linear_198 = None
        value_states_28 = view_86.transpose(1, 2)
        view_86 = None
        hidden_states_392 = query_states_56.to(torch.float32)
        query_states_56 = None
        pow_113 = hidden_states_392.pow(2)
        variance_112 = pow_113.mean(-1, keepdim=True)
        pow_113 = None
        add_224 = variance_112 + 1e-05
        variance_112 = None
        rsqrt_112 = torch.rsqrt(add_224)
        add_224 = None
        hidden_states_393 = hidden_states_392 * rsqrt_112
        hidden_states_392 = rsqrt_112 = None
        to_229 = hidden_states_393.to(torch.float16)
        hidden_states_393 = None
        query_states_57 = (
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_norm_parameters_weight_
            * to_229
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_229
        ) = None
        hidden_states_394 = key_states_56.to(torch.float32)
        key_states_56 = None
        pow_114 = hidden_states_394.pow(2)
        variance_113 = pow_114.mean(-1, keepdim=True)
        pow_114 = None
        add_225 = variance_113 + 1e-05
        variance_113 = None
        rsqrt_113 = torch.rsqrt(add_225)
        add_225 = None
        hidden_states_395 = hidden_states_394 * rsqrt_113
        hidden_states_394 = rsqrt_113 = None
        to_231 = hidden_states_395.to(torch.float16)
        hidden_states_395 = None
        key_states_57 = (
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_norm_parameters_weight_
            * to_231
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_231
        ) = None
        cos_31 = cos_2.unsqueeze(1)
        sin_31 = sin_2.unsqueeze(1)
        mul_370 = query_states_57 * cos_31
        x1_56 = query_states_57[(Ellipsis, slice(None, 32, None))]
        x2_56 = query_states_57[(Ellipsis, slice(32, None, None))]
        query_states_57 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_57 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_371 = cat_57 * sin_31
        cat_57 = None
        q_embed_28 = mul_370 + mul_371
        mul_370 = mul_371 = None
        mul_372 = key_states_57 * cos_31
        cos_31 = None
        x1_57 = key_states_57[(Ellipsis, slice(None, 32, None))]
        x2_57 = key_states_57[(Ellipsis, slice(32, None, None))]
        key_states_57 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_58 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_373 = cat_58 * sin_31
        cat_58 = sin_31 = None
        k_embed_28 = mul_372 + mul_373
        mul_372 = mul_373 = None
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
        hidden_states_396 = getitem_202.expand(1, 8, 4, 19, 64)
        getitem_202 = None
        key_56 = hidden_states_396.reshape(1, 32, 19, 64)
        hidden_states_396 = None
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
        hidden_states_397 = getitem_203.expand(1, 8, 4, 19, 64)
        getitem_203 = None
        value_56 = hidden_states_397.reshape(1, 32, 19, 64)
        hidden_states_397 = None
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
            scale=0.125,
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
        hidden_states_398 = attn_output_115.to(torch.float32)
        attn_output_115 = None
        pow_115 = hidden_states_398.pow(2)
        variance_114 = pow_115.mean(-1, keepdim=True)
        pow_115 = None
        add_228 = variance_114 + 1e-05
        variance_114 = None
        rsqrt_114 = torch.rsqrt(add_228)
        add_228 = None
        hidden_states_399 = hidden_states_398 * rsqrt_114
        hidden_states_398 = rsqrt_114 = None
        to_233 = hidden_states_399.to(torch.float16)
        hidden_states_399 = None
        hidden_states_400 = (
            l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
            * to_233
        )
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = (
            to_233
        ) = None
        hidden_states_401 = hidden_states_391 + hidden_states_400
        hidden_states_391 = hidden_states_400 = None
        linear_200 = torch._C._nn.linear(
            hidden_states_401,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_28 = torch.nn.functional.silu(linear_200, inplace=False)
        linear_200 = None
        linear_201 = torch._C._nn.linear(
            hidden_states_401,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_376 = silu_28 * linear_201
        silu_28 = linear_201 = None
        down_proj_28 = torch._C._nn.linear(
            mul_376,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_376 = l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_402 = down_proj_28.to(torch.float32)
        down_proj_28 = None
        pow_116 = hidden_states_402.pow(2)
        variance_115 = pow_116.mean(-1, keepdim=True)
        pow_116 = None
        add_230 = variance_115 + 1e-05
        variance_115 = None
        rsqrt_115 = torch.rsqrt(add_230)
        add_230 = None
        hidden_states_403 = hidden_states_402 * rsqrt_115
        hidden_states_402 = rsqrt_115 = None
        to_235 = hidden_states_403.to(torch.float16)
        hidden_states_403 = None
        hidden_states_404 = (
            l_self_modules_model_modules_layers_modules_28_modules_post_feedforward_layernorm_parameters_weight_
            * to_235
        )
        l_self_modules_model_modules_layers_modules_28_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_235
        ) = None
        hidden_states_405 = hidden_states_401 + hidden_states_404
        hidden_states_401 = hidden_states_404 = None
        linear_203 = torch._C._nn.linear(
            hidden_states_405,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        view_87 = linear_203.view((1, 19, -1, 64))
        linear_203 = None
        query_states_58 = view_87.transpose(1, 2)
        view_87 = None
        linear_204 = torch._C._nn.linear(
            hidden_states_405,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        view_88 = linear_204.view((1, 19, -1, 64))
        linear_204 = None
        key_states_58 = view_88.transpose(1, 2)
        view_88 = None
        linear_205 = torch._C._nn.linear(
            hidden_states_405,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = (
            None
        )
        view_89 = linear_205.view((1, 19, -1, 64))
        linear_205 = None
        value_states_29 = view_89.transpose(1, 2)
        view_89 = None
        hidden_states_406 = query_states_58.to(torch.float32)
        query_states_58 = None
        pow_117 = hidden_states_406.pow(2)
        variance_116 = pow_117.mean(-1, keepdim=True)
        pow_117 = None
        add_232 = variance_116 + 1e-05
        variance_116 = None
        rsqrt_116 = torch.rsqrt(add_232)
        add_232 = None
        hidden_states_407 = hidden_states_406 * rsqrt_116
        hidden_states_406 = rsqrt_116 = None
        to_237 = hidden_states_407.to(torch.float16)
        hidden_states_407 = None
        query_states_59 = (
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_norm_parameters_weight_
            * to_237
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_norm_parameters_weight_ = (
            to_237
        ) = None
        hidden_states_408 = key_states_58.to(torch.float32)
        key_states_58 = None
        pow_118 = hidden_states_408.pow(2)
        variance_117 = pow_118.mean(-1, keepdim=True)
        pow_118 = None
        add_233 = variance_117 + 1e-05
        variance_117 = None
        rsqrt_117 = torch.rsqrt(add_233)
        add_233 = None
        hidden_states_409 = hidden_states_408 * rsqrt_117
        hidden_states_408 = rsqrt_117 = None
        to_239 = hidden_states_409.to(torch.float16)
        hidden_states_409 = None
        key_states_59 = (
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_norm_parameters_weight_
            * to_239
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_norm_parameters_weight_ = (
            to_239
        ) = None
        cos_32 = cos_2.unsqueeze(1)
        cos_2 = None
        sin_32 = sin_2.unsqueeze(1)
        sin_2 = None
        mul_383 = query_states_59 * cos_32
        x1_58 = query_states_59[(Ellipsis, slice(None, 32, None))]
        x2_58 = query_states_59[(Ellipsis, slice(32, None, None))]
        query_states_59 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_59 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_384 = cat_59 * sin_32
        cat_59 = None
        q_embed_29 = mul_383 + mul_384
        mul_383 = mul_384 = None
        mul_385 = key_states_59 * cos_32
        cos_32 = None
        x1_59 = key_states_59[(Ellipsis, slice(None, 32, None))]
        x2_59 = key_states_59[(Ellipsis, slice(32, None, None))]
        key_states_59 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_60 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_386 = cat_60 * sin_32
        cat_60 = sin_32 = None
        k_embed_29 = mul_385 + mul_386
        mul_385 = mul_386 = None
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
        hidden_states_410 = getitem_209.expand(1, 8, 4, 19, 64)
        getitem_209 = None
        key_58 = hidden_states_410.reshape(1, 32, 19, 64)
        hidden_states_410 = None
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
        hidden_states_411 = getitem_210.expand(1, 8, 4, 19, 64)
        getitem_210 = None
        value_58 = hidden_states_411.reshape(1, 32, 19, 64)
        hidden_states_411 = None
        attention_mask_30 = causal_mask[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
            )
        ]
        causal_mask = None
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
            scale=0.125,
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
        hidden_states_412 = attn_output_119.to(torch.float32)
        attn_output_119 = None
        pow_119 = hidden_states_412.pow(2)
        variance_118 = pow_119.mean(-1, keepdim=True)
        pow_119 = None
        add_236 = variance_118 + 1e-05
        variance_118 = None
        rsqrt_118 = torch.rsqrt(add_236)
        add_236 = None
        hidden_states_413 = hidden_states_412 * rsqrt_118
        hidden_states_412 = rsqrt_118 = None
        to_241 = hidden_states_413.to(torch.float16)
        hidden_states_413 = None
        hidden_states_414 = (
            l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
            * to_241
        )
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = (
            to_241
        ) = None
        hidden_states_415 = hidden_states_405 + hidden_states_414
        hidden_states_405 = hidden_states_414 = None
        linear_207 = torch._C._nn.linear(
            hidden_states_415,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_29 = torch.nn.functional.silu(linear_207, inplace=False)
        linear_207 = None
        linear_208 = torch._C._nn.linear(
            hidden_states_415,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = (
            None
        )
        mul_389 = silu_29 * linear_208
        silu_29 = linear_208 = None
        down_proj_29 = torch._C._nn.linear(
            mul_389,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_389 = l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_416 = down_proj_29.to(torch.float32)
        down_proj_29 = None
        pow_120 = hidden_states_416.pow(2)
        variance_119 = pow_120.mean(-1, keepdim=True)
        pow_120 = None
        add_238 = variance_119 + 1e-05
        variance_119 = None
        rsqrt_119 = torch.rsqrt(add_238)
        add_238 = None
        hidden_states_417 = hidden_states_416 * rsqrt_119
        hidden_states_416 = rsqrt_119 = None
        to_243 = hidden_states_417.to(torch.float16)
        hidden_states_417 = None
        hidden_states_418 = (
            l_self_modules_model_modules_layers_modules_29_modules_post_feedforward_layernorm_parameters_weight_
            * to_243
        )
        l_self_modules_model_modules_layers_modules_29_modules_post_feedforward_layernorm_parameters_weight_ = (
            to_243
        ) = None
        hidden_states_419 = hidden_states_415 + hidden_states_418
        hidden_states_415 = hidden_states_418 = None
        hidden_states_420 = hidden_states_419.to(torch.float32)
        hidden_states_419 = None
        pow_121 = hidden_states_420.pow(2)
        variance_120 = pow_121.mean(-1, keepdim=True)
        pow_121 = None
        item_56 = l_self_modules_model_modules_norm_variance_epsilon.item()
        l_self_modules_model_modules_norm_variance_epsilon = None
        add_240 = variance_120 + item_56
        variance_120 = item_56 = None
        rsqrt_120 = torch.rsqrt(add_240)
        add_240 = None
        hidden_states_421 = hidden_states_420 * rsqrt_120
        hidden_states_420 = rsqrt_120 = None
        to_245 = hidden_states_421.to(torch.float16)
        hidden_states_421 = None
        hidden_states_422 = (
            l_self_modules_model_modules_norm_parameters_weight_ * to_245
        )
        l_self_modules_model_modules_norm_parameters_weight_ = to_245 = None
        getitem_212 = hidden_states_422[
            (slice(None, None, None), slice(0, None, None), slice(None, None, None))
        ]
        hidden_states_422 = None
        logits = torch._C._nn.linear(
            getitem_212,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
        )
        getitem_212 = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        return (logits,)
