import torch

from torch import device


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_input_ids_: torch.Tensor,
        L_self_modules_model_modules_embed_tokens_parameters_weight_: torch.nn.parameter.Parameter,
        L_attention_mask_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_cos_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_sin_cached_: torch.Tensor,
        L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_model_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_lm_head_parameters_weight_: torch.nn.parameter.Parameter,
    ):
        l_input_ids_ = L_input_ids_
        l_self_modules_model_modules_embed_tokens_parameters_weight_ = (
            L_self_modules_model_modules_embed_tokens_parameters_weight_
        )
        l_attention_mask_ = L_attention_mask_
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_cos_cached_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_sin_cached_
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_bias_ = L_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_bias_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_
        l_self_modules_model_modules_norm_parameters_weight_ = (
            L_self_modules_model_modules_norm_parameters_weight_
        )
        l_self_modules_model_modules_norm_parameters_bias_ = (
            L_self_modules_model_modules_norm_parameters_bias_
        )
        l_self_modules_lm_head_parameters_weight_ = (
            L_self_modules_lm_head_parameters_weight_
        )
        position_ids = torch.arange(0, 19, dtype=torch.int64, device=device(type="cpu"))
        unsqueeze = position_ids.unsqueeze(0)
        position_ids = None
        position_ids_1 = unsqueeze.view(-1, 19)
        unsqueeze = None
        inputs_embeds = torch.nn.functional.embedding(
            l_input_ids_,
            l_self_modules_model_modules_embed_tokens_parameters_weight_,
            None,
            None,
            2.0,
            False,
            False,
        )
        l_input_ids_ = (
            l_self_modules_model_modules_embed_tokens_parameters_weight_
        ) = None
        mask = torch.full((19, 19), -65504.0, device=device(type="cpu"))
        mask_cond = torch.arange(19, device=device(type="cpu"))
        add = mask_cond + 1
        view_1 = add.view(19, 1)
        add = None
        lt = mask_cond < view_1
        mask_cond = view_1 = None
        masked_fill_ = mask.masked_fill_(lt, 0)
        lt = masked_fill_ = None
        mask_1 = mask.to(torch.float16)
        mask = None
        getitem = mask_1[(None, None, slice(None, None, None), slice(None, None, None))]
        mask_1 = None
        combined_attention_mask = getitem.expand(1, 1, 19, 19)
        getitem = None
        getitem_1 = l_attention_mask_[
            (slice(None, None, None), None, None, slice(None, None, None))
        ]
        l_attention_mask_ = None
        expand_1 = getitem_1.expand(1, 1, 19, 19)
        getitem_1 = None
        expanded_mask = expand_1.to(torch.float16)
        expand_1 = None
        inverted_mask = 1.0 - expanded_mask
        expanded_mask = None
        to_2 = inverted_mask.to(torch.bool)
        masked_fill = inverted_mask.masked_fill(to_2, -65504.0)
        inverted_mask = to_2 = None
        expanded_attn_mask = masked_fill.to(device(type="cpu"))
        masked_fill = None
        combined_attention_mask_1 = expanded_attn_mask + combined_attention_mask
        expanded_attn_mask = combined_attention_mask = None
        hidden_states = torch.nn.functional.layer_norm(
            inputs_embeds,
            (2560,),
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_bias_ = (None)
        query_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states = torch._C._nn.linear(
            hidden_states,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_2 = query_states.view(1, 19, 32, 80)
        query_states = None
        query_states_1 = view_2.transpose(1, 2)
        view_2 = None
        view_3 = key_states.view(1, 19, 32, 80)
        key_states = None
        key_states_1 = view_3.transpose(1, 2)
        view_3 = None
        view_4 = value_states.view(1, 19, 32, 80)
        value_states = None
        value_states_1 = view_4.transpose(1, 2)
        view_4 = None
        query_rot = query_states_1[(Ellipsis, slice(None, 20, None))]
        query_pass = query_states_1[(Ellipsis, slice(20, None, None))]
        query_states_1 = None
        key_rot = key_states_1[(Ellipsis, slice(None, 20, None))]
        key_pass = key_states_1[(Ellipsis, slice(20, None, None))]
        key_states_1 = None
        getitem_6 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos = getitem_6.to(dtype=torch.float16)
        getitem_6 = None
        getitem_7 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin = getitem_7.to(dtype=torch.float16)
        getitem_7 = None
        squeeze = cos.squeeze(1)
        cos = None
        cos_1 = squeeze.squeeze(0)
        squeeze = None
        squeeze_2 = sin.squeeze(1)
        sin = None
        sin_1 = squeeze_2.squeeze(0)
        squeeze_2 = None
        getitem_8 = cos_1[position_ids_1]
        cos_1 = None
        cos_2 = getitem_8.unsqueeze(1)
        getitem_8 = None
        getitem_9 = sin_1[position_ids_1]
        sin_1 = None
        sin_2 = getitem_9.unsqueeze(1)
        getitem_9 = None
        mul = query_rot * cos_2
        chunk = torch.chunk(query_rot, 2, dim=-1)
        query_rot = None
        x1 = chunk[0]
        x2 = chunk[1]
        chunk = None
        neg = -x2
        x2 = None
        cat = torch.cat((neg, x1), dim=-1)
        neg = x1 = None
        mul_1 = cat * sin_2
        cat = None
        q_embed = mul + mul_1
        mul = mul_1 = None
        mul_2 = key_rot * cos_2
        cos_2 = None
        chunk_1 = torch.chunk(key_rot, 2, dim=-1)
        key_rot = None
        x1_1 = chunk_1[0]
        x2_1 = chunk_1[1]
        chunk_1 = None
        neg_1 = -x2_1
        x2_1 = None
        cat_1 = torch.cat((neg_1, x1_1), dim=-1)
        neg_1 = x1_1 = None
        mul_3 = cat_1 * sin_2
        cat_1 = sin_2 = None
        k_embed = mul_2 + mul_3
        mul_2 = mul_3 = None
        query_states_2 = torch.cat((q_embed, query_pass), dim=-1)
        q_embed = query_pass = None
        key_states_2 = torch.cat((k_embed, key_pass), dim=-1)
        k_embed = key_pass = None
        transpose_3 = key_states_2.transpose(2, 3)
        key_states_2 = None
        matmul = torch.matmul(query_states_2, transpose_3)
        query_states_2 = transpose_3 = None
        attn_weights = matmul / 8.94427190999916
        matmul = None
        attn_weights_1 = attn_weights + combined_attention_mask_1
        attn_weights = None
        softmax = torch.nn.functional.softmax(
            attn_weights_1, dim=-1, dtype=torch.float32
        )
        attn_weights_1 = None
        attn_weights_2 = softmax.to(torch.float16)
        softmax = None
        attn_output = torch.matmul(attn_weights_2, value_states_1)
        attn_weights_2 = value_states_1 = None
        transpose_4 = attn_output.transpose(1, 2)
        attn_output = None
        attn_output_1 = transpose_4.contiguous()
        transpose_4 = None
        attn_output_2 = attn_output_1.reshape(1, 19, 2560)
        attn_output_1 = None
        attn_output_3 = torch._C._nn.linear(
            attn_output_2,
            l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_2 = l_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_1 = inputs_embeds + attn_output_3
        inputs_embeds = attn_output_3 = None
        hidden_states_2 = torch.nn.functional.layer_norm(
            hidden_states_1,
            (2560,),
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_4 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu = torch.nn.functional.silu(linear_4, inplace=False)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            hidden_states_2,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_2 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_4 = silu * linear_5
        silu = linear_5 = None
        hidden_states_3 = torch._C._nn.linear(
            mul_4,
            l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_4 = l_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_4 = hidden_states_1 + hidden_states_3
        hidden_states_1 = hidden_states_3 = None
        hidden_states_5 = torch.nn.functional.layer_norm(
            hidden_states_4,
            (2560,),
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_bias_ = (None)
        query_states_3 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_3 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_2 = torch._C._nn.linear(
            hidden_states_5,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_5 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_5 = query_states_3.view(1, 19, 32, 80)
        query_states_3 = None
        query_states_4 = view_5.transpose(1, 2)
        view_5 = None
        view_6 = key_states_3.view(1, 19, 32, 80)
        key_states_3 = None
        key_states_4 = view_6.transpose(1, 2)
        view_6 = None
        view_7 = value_states_2.view(1, 19, 32, 80)
        value_states_2 = None
        value_states_3 = view_7.transpose(1, 2)
        view_7 = None
        query_rot_1 = query_states_4[(Ellipsis, slice(None, 20, None))]
        query_pass_1 = query_states_4[(Ellipsis, slice(20, None, None))]
        query_states_4 = None
        key_rot_1 = key_states_4[(Ellipsis, slice(None, 20, None))]
        key_pass_1 = key_states_4[(Ellipsis, slice(20, None, None))]
        key_states_4 = None
        getitem_18 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_3 = getitem_18.to(dtype=torch.float16)
        getitem_18 = None
        getitem_19 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_3 = getitem_19.to(dtype=torch.float16)
        getitem_19 = None
        squeeze_4 = cos_3.squeeze(1)
        cos_3 = None
        cos_4 = squeeze_4.squeeze(0)
        squeeze_4 = None
        squeeze_6 = sin_3.squeeze(1)
        sin_3 = None
        sin_4 = squeeze_6.squeeze(0)
        squeeze_6 = None
        getitem_20 = cos_4[position_ids_1]
        cos_4 = None
        cos_5 = getitem_20.unsqueeze(1)
        getitem_20 = None
        getitem_21 = sin_4[position_ids_1]
        sin_4 = None
        sin_5 = getitem_21.unsqueeze(1)
        getitem_21 = None
        mul_5 = query_rot_1 * cos_5
        chunk_2 = torch.chunk(query_rot_1, 2, dim=-1)
        query_rot_1 = None
        x1_2 = chunk_2[0]
        x2_2 = chunk_2[1]
        chunk_2 = None
        neg_2 = -x2_2
        x2_2 = None
        cat_4 = torch.cat((neg_2, x1_2), dim=-1)
        neg_2 = x1_2 = None
        mul_6 = cat_4 * sin_5
        cat_4 = None
        q_embed_1 = mul_5 + mul_6
        mul_5 = mul_6 = None
        mul_7 = key_rot_1 * cos_5
        cos_5 = None
        chunk_3 = torch.chunk(key_rot_1, 2, dim=-1)
        key_rot_1 = None
        x1_3 = chunk_3[0]
        x2_3 = chunk_3[1]
        chunk_3 = None
        neg_3 = -x2_3
        x2_3 = None
        cat_5 = torch.cat((neg_3, x1_3), dim=-1)
        neg_3 = x1_3 = None
        mul_8 = cat_5 * sin_5
        cat_5 = sin_5 = None
        k_embed_1 = mul_7 + mul_8
        mul_7 = mul_8 = None
        query_states_5 = torch.cat((q_embed_1, query_pass_1), dim=-1)
        q_embed_1 = query_pass_1 = None
        key_states_5 = torch.cat((k_embed_1, key_pass_1), dim=-1)
        k_embed_1 = key_pass_1 = None
        transpose_8 = key_states_5.transpose(2, 3)
        key_states_5 = None
        matmul_2 = torch.matmul(query_states_5, transpose_8)
        query_states_5 = transpose_8 = None
        attn_weights_3 = matmul_2 / 8.94427190999916
        matmul_2 = None
        attn_weights_4 = attn_weights_3 + combined_attention_mask_1
        attn_weights_3 = None
        softmax_1 = torch.nn.functional.softmax(
            attn_weights_4, dim=-1, dtype=torch.float32
        )
        attn_weights_4 = None
        attn_weights_5 = softmax_1.to(torch.float16)
        softmax_1 = None
        attn_output_4 = torch.matmul(attn_weights_5, value_states_3)
        attn_weights_5 = value_states_3 = None
        transpose_9 = attn_output_4.transpose(1, 2)
        attn_output_4 = None
        attn_output_5 = transpose_9.contiguous()
        transpose_9 = None
        attn_output_6 = attn_output_5.reshape(1, 19, 2560)
        attn_output_5 = None
        attn_output_7 = torch._C._nn.linear(
            attn_output_6,
            l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_6 = l_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_6 = hidden_states_4 + attn_output_7
        hidden_states_4 = attn_output_7 = None
        hidden_states_7 = torch.nn.functional.layer_norm(
            hidden_states_6,
            (2560,),
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_11 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_1 = torch.nn.functional.silu(linear_11, inplace=False)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            hidden_states_7,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_7 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_9 = silu_1 * linear_12
        silu_1 = linear_12 = None
        hidden_states_8 = torch._C._nn.linear(
            mul_9,
            l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_9 = l_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_9 = hidden_states_6 + hidden_states_8
        hidden_states_6 = hidden_states_8 = None
        hidden_states_10 = torch.nn.functional.layer_norm(
            hidden_states_9,
            (2560,),
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_bias_ = (None)
        query_states_6 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_6 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_4 = torch._C._nn.linear(
            hidden_states_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_8 = query_states_6.view(1, 19, 32, 80)
        query_states_6 = None
        query_states_7 = view_8.transpose(1, 2)
        view_8 = None
        view_9 = key_states_6.view(1, 19, 32, 80)
        key_states_6 = None
        key_states_7 = view_9.transpose(1, 2)
        view_9 = None
        view_10 = value_states_4.view(1, 19, 32, 80)
        value_states_4 = None
        value_states_5 = view_10.transpose(1, 2)
        view_10 = None
        query_rot_2 = query_states_7[(Ellipsis, slice(None, 20, None))]
        query_pass_2 = query_states_7[(Ellipsis, slice(20, None, None))]
        query_states_7 = None
        key_rot_2 = key_states_7[(Ellipsis, slice(None, 20, None))]
        key_pass_2 = key_states_7[(Ellipsis, slice(20, None, None))]
        key_states_7 = None
        getitem_30 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_6 = getitem_30.to(dtype=torch.float16)
        getitem_30 = None
        getitem_31 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_6 = getitem_31.to(dtype=torch.float16)
        getitem_31 = None
        squeeze_8 = cos_6.squeeze(1)
        cos_6 = None
        cos_7 = squeeze_8.squeeze(0)
        squeeze_8 = None
        squeeze_10 = sin_6.squeeze(1)
        sin_6 = None
        sin_7 = squeeze_10.squeeze(0)
        squeeze_10 = None
        getitem_32 = cos_7[position_ids_1]
        cos_7 = None
        cos_8 = getitem_32.unsqueeze(1)
        getitem_32 = None
        getitem_33 = sin_7[position_ids_1]
        sin_7 = None
        sin_8 = getitem_33.unsqueeze(1)
        getitem_33 = None
        mul_10 = query_rot_2 * cos_8
        chunk_4 = torch.chunk(query_rot_2, 2, dim=-1)
        query_rot_2 = None
        x1_4 = chunk_4[0]
        x2_4 = chunk_4[1]
        chunk_4 = None
        neg_4 = -x2_4
        x2_4 = None
        cat_8 = torch.cat((neg_4, x1_4), dim=-1)
        neg_4 = x1_4 = None
        mul_11 = cat_8 * sin_8
        cat_8 = None
        q_embed_2 = mul_10 + mul_11
        mul_10 = mul_11 = None
        mul_12 = key_rot_2 * cos_8
        cos_8 = None
        chunk_5 = torch.chunk(key_rot_2, 2, dim=-1)
        key_rot_2 = None
        x1_5 = chunk_5[0]
        x2_5 = chunk_5[1]
        chunk_5 = None
        neg_5 = -x2_5
        x2_5 = None
        cat_9 = torch.cat((neg_5, x1_5), dim=-1)
        neg_5 = x1_5 = None
        mul_13 = cat_9 * sin_8
        cat_9 = sin_8 = None
        k_embed_2 = mul_12 + mul_13
        mul_12 = mul_13 = None
        query_states_8 = torch.cat((q_embed_2, query_pass_2), dim=-1)
        q_embed_2 = query_pass_2 = None
        key_states_8 = torch.cat((k_embed_2, key_pass_2), dim=-1)
        k_embed_2 = key_pass_2 = None
        transpose_13 = key_states_8.transpose(2, 3)
        key_states_8 = None
        matmul_4 = torch.matmul(query_states_8, transpose_13)
        query_states_8 = transpose_13 = None
        attn_weights_6 = matmul_4 / 8.94427190999916
        matmul_4 = None
        attn_weights_7 = attn_weights_6 + combined_attention_mask_1
        attn_weights_6 = None
        softmax_2 = torch.nn.functional.softmax(
            attn_weights_7, dim=-1, dtype=torch.float32
        )
        attn_weights_7 = None
        attn_weights_8 = softmax_2.to(torch.float16)
        softmax_2 = None
        attn_output_8 = torch.matmul(attn_weights_8, value_states_5)
        attn_weights_8 = value_states_5 = None
        transpose_14 = attn_output_8.transpose(1, 2)
        attn_output_8 = None
        attn_output_9 = transpose_14.contiguous()
        transpose_14 = None
        attn_output_10 = attn_output_9.reshape(1, 19, 2560)
        attn_output_9 = None
        attn_output_11 = torch._C._nn.linear(
            attn_output_10,
            l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_10 = l_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_11 = hidden_states_9 + attn_output_11
        hidden_states_9 = attn_output_11 = None
        hidden_states_12 = torch.nn.functional.layer_norm(
            hidden_states_11,
            (2560,),
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_18 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_2 = torch.nn.functional.silu(linear_18, inplace=False)
        linear_18 = None
        linear_19 = torch._C._nn.linear(
            hidden_states_12,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_12 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_14 = silu_2 * linear_19
        silu_2 = linear_19 = None
        hidden_states_13 = torch._C._nn.linear(
            mul_14,
            l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_14 = l_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_14 = hidden_states_11 + hidden_states_13
        hidden_states_11 = hidden_states_13 = None
        hidden_states_15 = torch.nn.functional.layer_norm(
            hidden_states_14,
            (2560,),
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_bias_ = (None)
        query_states_9 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_9 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_6 = torch._C._nn.linear(
            hidden_states_15,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_15 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_11 = query_states_9.view(1, 19, 32, 80)
        query_states_9 = None
        query_states_10 = view_11.transpose(1, 2)
        view_11 = None
        view_12 = key_states_9.view(1, 19, 32, 80)
        key_states_9 = None
        key_states_10 = view_12.transpose(1, 2)
        view_12 = None
        view_13 = value_states_6.view(1, 19, 32, 80)
        value_states_6 = None
        value_states_7 = view_13.transpose(1, 2)
        view_13 = None
        query_rot_3 = query_states_10[(Ellipsis, slice(None, 20, None))]
        query_pass_3 = query_states_10[(Ellipsis, slice(20, None, None))]
        query_states_10 = None
        key_rot_3 = key_states_10[(Ellipsis, slice(None, 20, None))]
        key_pass_3 = key_states_10[(Ellipsis, slice(20, None, None))]
        key_states_10 = None
        getitem_42 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_9 = getitem_42.to(dtype=torch.float16)
        getitem_42 = None
        getitem_43 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_9 = getitem_43.to(dtype=torch.float16)
        getitem_43 = None
        squeeze_12 = cos_9.squeeze(1)
        cos_9 = None
        cos_10 = squeeze_12.squeeze(0)
        squeeze_12 = None
        squeeze_14 = sin_9.squeeze(1)
        sin_9 = None
        sin_10 = squeeze_14.squeeze(0)
        squeeze_14 = None
        getitem_44 = cos_10[position_ids_1]
        cos_10 = None
        cos_11 = getitem_44.unsqueeze(1)
        getitem_44 = None
        getitem_45 = sin_10[position_ids_1]
        sin_10 = None
        sin_11 = getitem_45.unsqueeze(1)
        getitem_45 = None
        mul_15 = query_rot_3 * cos_11
        chunk_6 = torch.chunk(query_rot_3, 2, dim=-1)
        query_rot_3 = None
        x1_6 = chunk_6[0]
        x2_6 = chunk_6[1]
        chunk_6 = None
        neg_6 = -x2_6
        x2_6 = None
        cat_12 = torch.cat((neg_6, x1_6), dim=-1)
        neg_6 = x1_6 = None
        mul_16 = cat_12 * sin_11
        cat_12 = None
        q_embed_3 = mul_15 + mul_16
        mul_15 = mul_16 = None
        mul_17 = key_rot_3 * cos_11
        cos_11 = None
        chunk_7 = torch.chunk(key_rot_3, 2, dim=-1)
        key_rot_3 = None
        x1_7 = chunk_7[0]
        x2_7 = chunk_7[1]
        chunk_7 = None
        neg_7 = -x2_7
        x2_7 = None
        cat_13 = torch.cat((neg_7, x1_7), dim=-1)
        neg_7 = x1_7 = None
        mul_18 = cat_13 * sin_11
        cat_13 = sin_11 = None
        k_embed_3 = mul_17 + mul_18
        mul_17 = mul_18 = None
        query_states_11 = torch.cat((q_embed_3, query_pass_3), dim=-1)
        q_embed_3 = query_pass_3 = None
        key_states_11 = torch.cat((k_embed_3, key_pass_3), dim=-1)
        k_embed_3 = key_pass_3 = None
        transpose_18 = key_states_11.transpose(2, 3)
        key_states_11 = None
        matmul_6 = torch.matmul(query_states_11, transpose_18)
        query_states_11 = transpose_18 = None
        attn_weights_9 = matmul_6 / 8.94427190999916
        matmul_6 = None
        attn_weights_10 = attn_weights_9 + combined_attention_mask_1
        attn_weights_9 = None
        softmax_3 = torch.nn.functional.softmax(
            attn_weights_10, dim=-1, dtype=torch.float32
        )
        attn_weights_10 = None
        attn_weights_11 = softmax_3.to(torch.float16)
        softmax_3 = None
        attn_output_12 = torch.matmul(attn_weights_11, value_states_7)
        attn_weights_11 = value_states_7 = None
        transpose_19 = attn_output_12.transpose(1, 2)
        attn_output_12 = None
        attn_output_13 = transpose_19.contiguous()
        transpose_19 = None
        attn_output_14 = attn_output_13.reshape(1, 19, 2560)
        attn_output_13 = None
        attn_output_15 = torch._C._nn.linear(
            attn_output_14,
            l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_14 = l_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_16 = hidden_states_14 + attn_output_15
        hidden_states_14 = attn_output_15 = None
        hidden_states_17 = torch.nn.functional.layer_norm(
            hidden_states_16,
            (2560,),
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_25 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_3 = torch.nn.functional.silu(linear_25, inplace=False)
        linear_25 = None
        linear_26 = torch._C._nn.linear(
            hidden_states_17,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_17 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_19 = silu_3 * linear_26
        silu_3 = linear_26 = None
        hidden_states_18 = torch._C._nn.linear(
            mul_19,
            l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_19 = l_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_19 = hidden_states_16 + hidden_states_18
        hidden_states_16 = hidden_states_18 = None
        hidden_states_20 = torch.nn.functional.layer_norm(
            hidden_states_19,
            (2560,),
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_bias_ = (None)
        query_states_12 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_12 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_8 = torch._C._nn.linear(
            hidden_states_20,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_20 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_14 = query_states_12.view(1, 19, 32, 80)
        query_states_12 = None
        query_states_13 = view_14.transpose(1, 2)
        view_14 = None
        view_15 = key_states_12.view(1, 19, 32, 80)
        key_states_12 = None
        key_states_13 = view_15.transpose(1, 2)
        view_15 = None
        view_16 = value_states_8.view(1, 19, 32, 80)
        value_states_8 = None
        value_states_9 = view_16.transpose(1, 2)
        view_16 = None
        query_rot_4 = query_states_13[(Ellipsis, slice(None, 20, None))]
        query_pass_4 = query_states_13[(Ellipsis, slice(20, None, None))]
        query_states_13 = None
        key_rot_4 = key_states_13[(Ellipsis, slice(None, 20, None))]
        key_pass_4 = key_states_13[(Ellipsis, slice(20, None, None))]
        key_states_13 = None
        getitem_54 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_12 = getitem_54.to(dtype=torch.float16)
        getitem_54 = None
        getitem_55 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_12 = getitem_55.to(dtype=torch.float16)
        getitem_55 = None
        squeeze_16 = cos_12.squeeze(1)
        cos_12 = None
        cos_13 = squeeze_16.squeeze(0)
        squeeze_16 = None
        squeeze_18 = sin_12.squeeze(1)
        sin_12 = None
        sin_13 = squeeze_18.squeeze(0)
        squeeze_18 = None
        getitem_56 = cos_13[position_ids_1]
        cos_13 = None
        cos_14 = getitem_56.unsqueeze(1)
        getitem_56 = None
        getitem_57 = sin_13[position_ids_1]
        sin_13 = None
        sin_14 = getitem_57.unsqueeze(1)
        getitem_57 = None
        mul_20 = query_rot_4 * cos_14
        chunk_8 = torch.chunk(query_rot_4, 2, dim=-1)
        query_rot_4 = None
        x1_8 = chunk_8[0]
        x2_8 = chunk_8[1]
        chunk_8 = None
        neg_8 = -x2_8
        x2_8 = None
        cat_16 = torch.cat((neg_8, x1_8), dim=-1)
        neg_8 = x1_8 = None
        mul_21 = cat_16 * sin_14
        cat_16 = None
        q_embed_4 = mul_20 + mul_21
        mul_20 = mul_21 = None
        mul_22 = key_rot_4 * cos_14
        cos_14 = None
        chunk_9 = torch.chunk(key_rot_4, 2, dim=-1)
        key_rot_4 = None
        x1_9 = chunk_9[0]
        x2_9 = chunk_9[1]
        chunk_9 = None
        neg_9 = -x2_9
        x2_9 = None
        cat_17 = torch.cat((neg_9, x1_9), dim=-1)
        neg_9 = x1_9 = None
        mul_23 = cat_17 * sin_14
        cat_17 = sin_14 = None
        k_embed_4 = mul_22 + mul_23
        mul_22 = mul_23 = None
        query_states_14 = torch.cat((q_embed_4, query_pass_4), dim=-1)
        q_embed_4 = query_pass_4 = None
        key_states_14 = torch.cat((k_embed_4, key_pass_4), dim=-1)
        k_embed_4 = key_pass_4 = None
        transpose_23 = key_states_14.transpose(2, 3)
        key_states_14 = None
        matmul_8 = torch.matmul(query_states_14, transpose_23)
        query_states_14 = transpose_23 = None
        attn_weights_12 = matmul_8 / 8.94427190999916
        matmul_8 = None
        attn_weights_13 = attn_weights_12 + combined_attention_mask_1
        attn_weights_12 = None
        softmax_4 = torch.nn.functional.softmax(
            attn_weights_13, dim=-1, dtype=torch.float32
        )
        attn_weights_13 = None
        attn_weights_14 = softmax_4.to(torch.float16)
        softmax_4 = None
        attn_output_16 = torch.matmul(attn_weights_14, value_states_9)
        attn_weights_14 = value_states_9 = None
        transpose_24 = attn_output_16.transpose(1, 2)
        attn_output_16 = None
        attn_output_17 = transpose_24.contiguous()
        transpose_24 = None
        attn_output_18 = attn_output_17.reshape(1, 19, 2560)
        attn_output_17 = None
        attn_output_19 = torch._C._nn.linear(
            attn_output_18,
            l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_18 = l_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_21 = hidden_states_19 + attn_output_19
        hidden_states_19 = attn_output_19 = None
        hidden_states_22 = torch.nn.functional.layer_norm(
            hidden_states_21,
            (2560,),
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_32 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_4 = torch.nn.functional.silu(linear_32, inplace=False)
        linear_32 = None
        linear_33 = torch._C._nn.linear(
            hidden_states_22,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_22 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_24 = silu_4 * linear_33
        silu_4 = linear_33 = None
        hidden_states_23 = torch._C._nn.linear(
            mul_24,
            l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_24 = l_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_24 = hidden_states_21 + hidden_states_23
        hidden_states_21 = hidden_states_23 = None
        hidden_states_25 = torch.nn.functional.layer_norm(
            hidden_states_24,
            (2560,),
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_bias_ = (None)
        query_states_15 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_15 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_10 = torch._C._nn.linear(
            hidden_states_25,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_25 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_17 = query_states_15.view(1, 19, 32, 80)
        query_states_15 = None
        query_states_16 = view_17.transpose(1, 2)
        view_17 = None
        view_18 = key_states_15.view(1, 19, 32, 80)
        key_states_15 = None
        key_states_16 = view_18.transpose(1, 2)
        view_18 = None
        view_19 = value_states_10.view(1, 19, 32, 80)
        value_states_10 = None
        value_states_11 = view_19.transpose(1, 2)
        view_19 = None
        query_rot_5 = query_states_16[(Ellipsis, slice(None, 20, None))]
        query_pass_5 = query_states_16[(Ellipsis, slice(20, None, None))]
        query_states_16 = None
        key_rot_5 = key_states_16[(Ellipsis, slice(None, 20, None))]
        key_pass_5 = key_states_16[(Ellipsis, slice(20, None, None))]
        key_states_16 = None
        getitem_66 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_15 = getitem_66.to(dtype=torch.float16)
        getitem_66 = None
        getitem_67 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_15 = getitem_67.to(dtype=torch.float16)
        getitem_67 = None
        squeeze_20 = cos_15.squeeze(1)
        cos_15 = None
        cos_16 = squeeze_20.squeeze(0)
        squeeze_20 = None
        squeeze_22 = sin_15.squeeze(1)
        sin_15 = None
        sin_16 = squeeze_22.squeeze(0)
        squeeze_22 = None
        getitem_68 = cos_16[position_ids_1]
        cos_16 = None
        cos_17 = getitem_68.unsqueeze(1)
        getitem_68 = None
        getitem_69 = sin_16[position_ids_1]
        sin_16 = None
        sin_17 = getitem_69.unsqueeze(1)
        getitem_69 = None
        mul_25 = query_rot_5 * cos_17
        chunk_10 = torch.chunk(query_rot_5, 2, dim=-1)
        query_rot_5 = None
        x1_10 = chunk_10[0]
        x2_10 = chunk_10[1]
        chunk_10 = None
        neg_10 = -x2_10
        x2_10 = None
        cat_20 = torch.cat((neg_10, x1_10), dim=-1)
        neg_10 = x1_10 = None
        mul_26 = cat_20 * sin_17
        cat_20 = None
        q_embed_5 = mul_25 + mul_26
        mul_25 = mul_26 = None
        mul_27 = key_rot_5 * cos_17
        cos_17 = None
        chunk_11 = torch.chunk(key_rot_5, 2, dim=-1)
        key_rot_5 = None
        x1_11 = chunk_11[0]
        x2_11 = chunk_11[1]
        chunk_11 = None
        neg_11 = -x2_11
        x2_11 = None
        cat_21 = torch.cat((neg_11, x1_11), dim=-1)
        neg_11 = x1_11 = None
        mul_28 = cat_21 * sin_17
        cat_21 = sin_17 = None
        k_embed_5 = mul_27 + mul_28
        mul_27 = mul_28 = None
        query_states_17 = torch.cat((q_embed_5, query_pass_5), dim=-1)
        q_embed_5 = query_pass_5 = None
        key_states_17 = torch.cat((k_embed_5, key_pass_5), dim=-1)
        k_embed_5 = key_pass_5 = None
        transpose_28 = key_states_17.transpose(2, 3)
        key_states_17 = None
        matmul_10 = torch.matmul(query_states_17, transpose_28)
        query_states_17 = transpose_28 = None
        attn_weights_15 = matmul_10 / 8.94427190999916
        matmul_10 = None
        attn_weights_16 = attn_weights_15 + combined_attention_mask_1
        attn_weights_15 = None
        softmax_5 = torch.nn.functional.softmax(
            attn_weights_16, dim=-1, dtype=torch.float32
        )
        attn_weights_16 = None
        attn_weights_17 = softmax_5.to(torch.float16)
        softmax_5 = None
        attn_output_20 = torch.matmul(attn_weights_17, value_states_11)
        attn_weights_17 = value_states_11 = None
        transpose_29 = attn_output_20.transpose(1, 2)
        attn_output_20 = None
        attn_output_21 = transpose_29.contiguous()
        transpose_29 = None
        attn_output_22 = attn_output_21.reshape(1, 19, 2560)
        attn_output_21 = None
        attn_output_23 = torch._C._nn.linear(
            attn_output_22,
            l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_22 = l_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_26 = hidden_states_24 + attn_output_23
        hidden_states_24 = attn_output_23 = None
        hidden_states_27 = torch.nn.functional.layer_norm(
            hidden_states_26,
            (2560,),
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_39 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_5 = torch.nn.functional.silu(linear_39, inplace=False)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            hidden_states_27,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_27 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_29 = silu_5 * linear_40
        silu_5 = linear_40 = None
        hidden_states_28 = torch._C._nn.linear(
            mul_29,
            l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_29 = l_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_29 = hidden_states_26 + hidden_states_28
        hidden_states_26 = hidden_states_28 = None
        hidden_states_30 = torch.nn.functional.layer_norm(
            hidden_states_29,
            (2560,),
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_bias_ = (None)
        query_states_18 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_18 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_12 = torch._C._nn.linear(
            hidden_states_30,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_30 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_20 = query_states_18.view(1, 19, 32, 80)
        query_states_18 = None
        query_states_19 = view_20.transpose(1, 2)
        view_20 = None
        view_21 = key_states_18.view(1, 19, 32, 80)
        key_states_18 = None
        key_states_19 = view_21.transpose(1, 2)
        view_21 = None
        view_22 = value_states_12.view(1, 19, 32, 80)
        value_states_12 = None
        value_states_13 = view_22.transpose(1, 2)
        view_22 = None
        query_rot_6 = query_states_19[(Ellipsis, slice(None, 20, None))]
        query_pass_6 = query_states_19[(Ellipsis, slice(20, None, None))]
        query_states_19 = None
        key_rot_6 = key_states_19[(Ellipsis, slice(None, 20, None))]
        key_pass_6 = key_states_19[(Ellipsis, slice(20, None, None))]
        key_states_19 = None
        getitem_78 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_18 = getitem_78.to(dtype=torch.float16)
        getitem_78 = None
        getitem_79 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_18 = getitem_79.to(dtype=torch.float16)
        getitem_79 = None
        squeeze_24 = cos_18.squeeze(1)
        cos_18 = None
        cos_19 = squeeze_24.squeeze(0)
        squeeze_24 = None
        squeeze_26 = sin_18.squeeze(1)
        sin_18 = None
        sin_19 = squeeze_26.squeeze(0)
        squeeze_26 = None
        getitem_80 = cos_19[position_ids_1]
        cos_19 = None
        cos_20 = getitem_80.unsqueeze(1)
        getitem_80 = None
        getitem_81 = sin_19[position_ids_1]
        sin_19 = None
        sin_20 = getitem_81.unsqueeze(1)
        getitem_81 = None
        mul_30 = query_rot_6 * cos_20
        chunk_12 = torch.chunk(query_rot_6, 2, dim=-1)
        query_rot_6 = None
        x1_12 = chunk_12[0]
        x2_12 = chunk_12[1]
        chunk_12 = None
        neg_12 = -x2_12
        x2_12 = None
        cat_24 = torch.cat((neg_12, x1_12), dim=-1)
        neg_12 = x1_12 = None
        mul_31 = cat_24 * sin_20
        cat_24 = None
        q_embed_6 = mul_30 + mul_31
        mul_30 = mul_31 = None
        mul_32 = key_rot_6 * cos_20
        cos_20 = None
        chunk_13 = torch.chunk(key_rot_6, 2, dim=-1)
        key_rot_6 = None
        x1_13 = chunk_13[0]
        x2_13 = chunk_13[1]
        chunk_13 = None
        neg_13 = -x2_13
        x2_13 = None
        cat_25 = torch.cat((neg_13, x1_13), dim=-1)
        neg_13 = x1_13 = None
        mul_33 = cat_25 * sin_20
        cat_25 = sin_20 = None
        k_embed_6 = mul_32 + mul_33
        mul_32 = mul_33 = None
        query_states_20 = torch.cat((q_embed_6, query_pass_6), dim=-1)
        q_embed_6 = query_pass_6 = None
        key_states_20 = torch.cat((k_embed_6, key_pass_6), dim=-1)
        k_embed_6 = key_pass_6 = None
        transpose_33 = key_states_20.transpose(2, 3)
        key_states_20 = None
        matmul_12 = torch.matmul(query_states_20, transpose_33)
        query_states_20 = transpose_33 = None
        attn_weights_18 = matmul_12 / 8.94427190999916
        matmul_12 = None
        attn_weights_19 = attn_weights_18 + combined_attention_mask_1
        attn_weights_18 = None
        softmax_6 = torch.nn.functional.softmax(
            attn_weights_19, dim=-1, dtype=torch.float32
        )
        attn_weights_19 = None
        attn_weights_20 = softmax_6.to(torch.float16)
        softmax_6 = None
        attn_output_24 = torch.matmul(attn_weights_20, value_states_13)
        attn_weights_20 = value_states_13 = None
        transpose_34 = attn_output_24.transpose(1, 2)
        attn_output_24 = None
        attn_output_25 = transpose_34.contiguous()
        transpose_34 = None
        attn_output_26 = attn_output_25.reshape(1, 19, 2560)
        attn_output_25 = None
        attn_output_27 = torch._C._nn.linear(
            attn_output_26,
            l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_26 = l_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_31 = hidden_states_29 + attn_output_27
        hidden_states_29 = attn_output_27 = None
        hidden_states_32 = torch.nn.functional.layer_norm(
            hidden_states_31,
            (2560,),
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_46 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_6 = torch.nn.functional.silu(linear_46, inplace=False)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            hidden_states_32,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_32 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_34 = silu_6 * linear_47
        silu_6 = linear_47 = None
        hidden_states_33 = torch._C._nn.linear(
            mul_34,
            l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_34 = l_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_34 = hidden_states_31 + hidden_states_33
        hidden_states_31 = hidden_states_33 = None
        hidden_states_35 = torch.nn.functional.layer_norm(
            hidden_states_34,
            (2560,),
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_bias_ = (None)
        query_states_21 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_21 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_14 = torch._C._nn.linear(
            hidden_states_35,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_35 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_23 = query_states_21.view(1, 19, 32, 80)
        query_states_21 = None
        query_states_22 = view_23.transpose(1, 2)
        view_23 = None
        view_24 = key_states_21.view(1, 19, 32, 80)
        key_states_21 = None
        key_states_22 = view_24.transpose(1, 2)
        view_24 = None
        view_25 = value_states_14.view(1, 19, 32, 80)
        value_states_14 = None
        value_states_15 = view_25.transpose(1, 2)
        view_25 = None
        query_rot_7 = query_states_22[(Ellipsis, slice(None, 20, None))]
        query_pass_7 = query_states_22[(Ellipsis, slice(20, None, None))]
        query_states_22 = None
        key_rot_7 = key_states_22[(Ellipsis, slice(None, 20, None))]
        key_pass_7 = key_states_22[(Ellipsis, slice(20, None, None))]
        key_states_22 = None
        getitem_90 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_21 = getitem_90.to(dtype=torch.float16)
        getitem_90 = None
        getitem_91 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_21 = getitem_91.to(dtype=torch.float16)
        getitem_91 = None
        squeeze_28 = cos_21.squeeze(1)
        cos_21 = None
        cos_22 = squeeze_28.squeeze(0)
        squeeze_28 = None
        squeeze_30 = sin_21.squeeze(1)
        sin_21 = None
        sin_22 = squeeze_30.squeeze(0)
        squeeze_30 = None
        getitem_92 = cos_22[position_ids_1]
        cos_22 = None
        cos_23 = getitem_92.unsqueeze(1)
        getitem_92 = None
        getitem_93 = sin_22[position_ids_1]
        sin_22 = None
        sin_23 = getitem_93.unsqueeze(1)
        getitem_93 = None
        mul_35 = query_rot_7 * cos_23
        chunk_14 = torch.chunk(query_rot_7, 2, dim=-1)
        query_rot_7 = None
        x1_14 = chunk_14[0]
        x2_14 = chunk_14[1]
        chunk_14 = None
        neg_14 = -x2_14
        x2_14 = None
        cat_28 = torch.cat((neg_14, x1_14), dim=-1)
        neg_14 = x1_14 = None
        mul_36 = cat_28 * sin_23
        cat_28 = None
        q_embed_7 = mul_35 + mul_36
        mul_35 = mul_36 = None
        mul_37 = key_rot_7 * cos_23
        cos_23 = None
        chunk_15 = torch.chunk(key_rot_7, 2, dim=-1)
        key_rot_7 = None
        x1_15 = chunk_15[0]
        x2_15 = chunk_15[1]
        chunk_15 = None
        neg_15 = -x2_15
        x2_15 = None
        cat_29 = torch.cat((neg_15, x1_15), dim=-1)
        neg_15 = x1_15 = None
        mul_38 = cat_29 * sin_23
        cat_29 = sin_23 = None
        k_embed_7 = mul_37 + mul_38
        mul_37 = mul_38 = None
        query_states_23 = torch.cat((q_embed_7, query_pass_7), dim=-1)
        q_embed_7 = query_pass_7 = None
        key_states_23 = torch.cat((k_embed_7, key_pass_7), dim=-1)
        k_embed_7 = key_pass_7 = None
        transpose_38 = key_states_23.transpose(2, 3)
        key_states_23 = None
        matmul_14 = torch.matmul(query_states_23, transpose_38)
        query_states_23 = transpose_38 = None
        attn_weights_21 = matmul_14 / 8.94427190999916
        matmul_14 = None
        attn_weights_22 = attn_weights_21 + combined_attention_mask_1
        attn_weights_21 = None
        softmax_7 = torch.nn.functional.softmax(
            attn_weights_22, dim=-1, dtype=torch.float32
        )
        attn_weights_22 = None
        attn_weights_23 = softmax_7.to(torch.float16)
        softmax_7 = None
        attn_output_28 = torch.matmul(attn_weights_23, value_states_15)
        attn_weights_23 = value_states_15 = None
        transpose_39 = attn_output_28.transpose(1, 2)
        attn_output_28 = None
        attn_output_29 = transpose_39.contiguous()
        transpose_39 = None
        attn_output_30 = attn_output_29.reshape(1, 19, 2560)
        attn_output_29 = None
        attn_output_31 = torch._C._nn.linear(
            attn_output_30,
            l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_30 = l_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_36 = hidden_states_34 + attn_output_31
        hidden_states_34 = attn_output_31 = None
        hidden_states_37 = torch.nn.functional.layer_norm(
            hidden_states_36,
            (2560,),
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_53 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_7 = torch.nn.functional.silu(linear_53, inplace=False)
        linear_53 = None
        linear_54 = torch._C._nn.linear(
            hidden_states_37,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_37 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_39 = silu_7 * linear_54
        silu_7 = linear_54 = None
        hidden_states_38 = torch._C._nn.linear(
            mul_39,
            l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_39 = l_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_39 = hidden_states_36 + hidden_states_38
        hidden_states_36 = hidden_states_38 = None
        hidden_states_40 = torch.nn.functional.layer_norm(
            hidden_states_39,
            (2560,),
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_bias_ = (None)
        query_states_24 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_24 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_16 = torch._C._nn.linear(
            hidden_states_40,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_40 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_26 = query_states_24.view(1, 19, 32, 80)
        query_states_24 = None
        query_states_25 = view_26.transpose(1, 2)
        view_26 = None
        view_27 = key_states_24.view(1, 19, 32, 80)
        key_states_24 = None
        key_states_25 = view_27.transpose(1, 2)
        view_27 = None
        view_28 = value_states_16.view(1, 19, 32, 80)
        value_states_16 = None
        value_states_17 = view_28.transpose(1, 2)
        view_28 = None
        query_rot_8 = query_states_25[(Ellipsis, slice(None, 20, None))]
        query_pass_8 = query_states_25[(Ellipsis, slice(20, None, None))]
        query_states_25 = None
        key_rot_8 = key_states_25[(Ellipsis, slice(None, 20, None))]
        key_pass_8 = key_states_25[(Ellipsis, slice(20, None, None))]
        key_states_25 = None
        getitem_102 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_24 = getitem_102.to(dtype=torch.float16)
        getitem_102 = None
        getitem_103 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_24 = getitem_103.to(dtype=torch.float16)
        getitem_103 = None
        squeeze_32 = cos_24.squeeze(1)
        cos_24 = None
        cos_25 = squeeze_32.squeeze(0)
        squeeze_32 = None
        squeeze_34 = sin_24.squeeze(1)
        sin_24 = None
        sin_25 = squeeze_34.squeeze(0)
        squeeze_34 = None
        getitem_104 = cos_25[position_ids_1]
        cos_25 = None
        cos_26 = getitem_104.unsqueeze(1)
        getitem_104 = None
        getitem_105 = sin_25[position_ids_1]
        sin_25 = None
        sin_26 = getitem_105.unsqueeze(1)
        getitem_105 = None
        mul_40 = query_rot_8 * cos_26
        chunk_16 = torch.chunk(query_rot_8, 2, dim=-1)
        query_rot_8 = None
        x1_16 = chunk_16[0]
        x2_16 = chunk_16[1]
        chunk_16 = None
        neg_16 = -x2_16
        x2_16 = None
        cat_32 = torch.cat((neg_16, x1_16), dim=-1)
        neg_16 = x1_16 = None
        mul_41 = cat_32 * sin_26
        cat_32 = None
        q_embed_8 = mul_40 + mul_41
        mul_40 = mul_41 = None
        mul_42 = key_rot_8 * cos_26
        cos_26 = None
        chunk_17 = torch.chunk(key_rot_8, 2, dim=-1)
        key_rot_8 = None
        x1_17 = chunk_17[0]
        x2_17 = chunk_17[1]
        chunk_17 = None
        neg_17 = -x2_17
        x2_17 = None
        cat_33 = torch.cat((neg_17, x1_17), dim=-1)
        neg_17 = x1_17 = None
        mul_43 = cat_33 * sin_26
        cat_33 = sin_26 = None
        k_embed_8 = mul_42 + mul_43
        mul_42 = mul_43 = None
        query_states_26 = torch.cat((q_embed_8, query_pass_8), dim=-1)
        q_embed_8 = query_pass_8 = None
        key_states_26 = torch.cat((k_embed_8, key_pass_8), dim=-1)
        k_embed_8 = key_pass_8 = None
        transpose_43 = key_states_26.transpose(2, 3)
        key_states_26 = None
        matmul_16 = torch.matmul(query_states_26, transpose_43)
        query_states_26 = transpose_43 = None
        attn_weights_24 = matmul_16 / 8.94427190999916
        matmul_16 = None
        attn_weights_25 = attn_weights_24 + combined_attention_mask_1
        attn_weights_24 = None
        softmax_8 = torch.nn.functional.softmax(
            attn_weights_25, dim=-1, dtype=torch.float32
        )
        attn_weights_25 = None
        attn_weights_26 = softmax_8.to(torch.float16)
        softmax_8 = None
        attn_output_32 = torch.matmul(attn_weights_26, value_states_17)
        attn_weights_26 = value_states_17 = None
        transpose_44 = attn_output_32.transpose(1, 2)
        attn_output_32 = None
        attn_output_33 = transpose_44.contiguous()
        transpose_44 = None
        attn_output_34 = attn_output_33.reshape(1, 19, 2560)
        attn_output_33 = None
        attn_output_35 = torch._C._nn.linear(
            attn_output_34,
            l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_34 = l_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_41 = hidden_states_39 + attn_output_35
        hidden_states_39 = attn_output_35 = None
        hidden_states_42 = torch.nn.functional.layer_norm(
            hidden_states_41,
            (2560,),
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_60 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_8 = torch.nn.functional.silu(linear_60, inplace=False)
        linear_60 = None
        linear_61 = torch._C._nn.linear(
            hidden_states_42,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_42 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_44 = silu_8 * linear_61
        silu_8 = linear_61 = None
        hidden_states_43 = torch._C._nn.linear(
            mul_44,
            l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_44 = l_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_44 = hidden_states_41 + hidden_states_43
        hidden_states_41 = hidden_states_43 = None
        hidden_states_45 = torch.nn.functional.layer_norm(
            hidden_states_44,
            (2560,),
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_bias_ = (None)
        query_states_27 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_27 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_18 = torch._C._nn.linear(
            hidden_states_45,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_45 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_29 = query_states_27.view(1, 19, 32, 80)
        query_states_27 = None
        query_states_28 = view_29.transpose(1, 2)
        view_29 = None
        view_30 = key_states_27.view(1, 19, 32, 80)
        key_states_27 = None
        key_states_28 = view_30.transpose(1, 2)
        view_30 = None
        view_31 = value_states_18.view(1, 19, 32, 80)
        value_states_18 = None
        value_states_19 = view_31.transpose(1, 2)
        view_31 = None
        query_rot_9 = query_states_28[(Ellipsis, slice(None, 20, None))]
        query_pass_9 = query_states_28[(Ellipsis, slice(20, None, None))]
        query_states_28 = None
        key_rot_9 = key_states_28[(Ellipsis, slice(None, 20, None))]
        key_pass_9 = key_states_28[(Ellipsis, slice(20, None, None))]
        key_states_28 = None
        getitem_114 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_27 = getitem_114.to(dtype=torch.float16)
        getitem_114 = None
        getitem_115 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_27 = getitem_115.to(dtype=torch.float16)
        getitem_115 = None
        squeeze_36 = cos_27.squeeze(1)
        cos_27 = None
        cos_28 = squeeze_36.squeeze(0)
        squeeze_36 = None
        squeeze_38 = sin_27.squeeze(1)
        sin_27 = None
        sin_28 = squeeze_38.squeeze(0)
        squeeze_38 = None
        getitem_116 = cos_28[position_ids_1]
        cos_28 = None
        cos_29 = getitem_116.unsqueeze(1)
        getitem_116 = None
        getitem_117 = sin_28[position_ids_1]
        sin_28 = None
        sin_29 = getitem_117.unsqueeze(1)
        getitem_117 = None
        mul_45 = query_rot_9 * cos_29
        chunk_18 = torch.chunk(query_rot_9, 2, dim=-1)
        query_rot_9 = None
        x1_18 = chunk_18[0]
        x2_18 = chunk_18[1]
        chunk_18 = None
        neg_18 = -x2_18
        x2_18 = None
        cat_36 = torch.cat((neg_18, x1_18), dim=-1)
        neg_18 = x1_18 = None
        mul_46 = cat_36 * sin_29
        cat_36 = None
        q_embed_9 = mul_45 + mul_46
        mul_45 = mul_46 = None
        mul_47 = key_rot_9 * cos_29
        cos_29 = None
        chunk_19 = torch.chunk(key_rot_9, 2, dim=-1)
        key_rot_9 = None
        x1_19 = chunk_19[0]
        x2_19 = chunk_19[1]
        chunk_19 = None
        neg_19 = -x2_19
        x2_19 = None
        cat_37 = torch.cat((neg_19, x1_19), dim=-1)
        neg_19 = x1_19 = None
        mul_48 = cat_37 * sin_29
        cat_37 = sin_29 = None
        k_embed_9 = mul_47 + mul_48
        mul_47 = mul_48 = None
        query_states_29 = torch.cat((q_embed_9, query_pass_9), dim=-1)
        q_embed_9 = query_pass_9 = None
        key_states_29 = torch.cat((k_embed_9, key_pass_9), dim=-1)
        k_embed_9 = key_pass_9 = None
        transpose_48 = key_states_29.transpose(2, 3)
        key_states_29 = None
        matmul_18 = torch.matmul(query_states_29, transpose_48)
        query_states_29 = transpose_48 = None
        attn_weights_27 = matmul_18 / 8.94427190999916
        matmul_18 = None
        attn_weights_28 = attn_weights_27 + combined_attention_mask_1
        attn_weights_27 = None
        softmax_9 = torch.nn.functional.softmax(
            attn_weights_28, dim=-1, dtype=torch.float32
        )
        attn_weights_28 = None
        attn_weights_29 = softmax_9.to(torch.float16)
        softmax_9 = None
        attn_output_36 = torch.matmul(attn_weights_29, value_states_19)
        attn_weights_29 = value_states_19 = None
        transpose_49 = attn_output_36.transpose(1, 2)
        attn_output_36 = None
        attn_output_37 = transpose_49.contiguous()
        transpose_49 = None
        attn_output_38 = attn_output_37.reshape(1, 19, 2560)
        attn_output_37 = None
        attn_output_39 = torch._C._nn.linear(
            attn_output_38,
            l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_38 = l_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_46 = hidden_states_44 + attn_output_39
        hidden_states_44 = attn_output_39 = None
        hidden_states_47 = torch.nn.functional.layer_norm(
            hidden_states_46,
            (2560,),
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_67 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_9 = torch.nn.functional.silu(linear_67, inplace=False)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            hidden_states_47,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_47 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_49 = silu_9 * linear_68
        silu_9 = linear_68 = None
        hidden_states_48 = torch._C._nn.linear(
            mul_49,
            l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_49 = l_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_49 = hidden_states_46 + hidden_states_48
        hidden_states_46 = hidden_states_48 = None
        hidden_states_50 = torch.nn.functional.layer_norm(
            hidden_states_49,
            (2560,),
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_bias_ = (None)
        query_states_30 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_30 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_20 = torch._C._nn.linear(
            hidden_states_50,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_50 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_32 = query_states_30.view(1, 19, 32, 80)
        query_states_30 = None
        query_states_31 = view_32.transpose(1, 2)
        view_32 = None
        view_33 = key_states_30.view(1, 19, 32, 80)
        key_states_30 = None
        key_states_31 = view_33.transpose(1, 2)
        view_33 = None
        view_34 = value_states_20.view(1, 19, 32, 80)
        value_states_20 = None
        value_states_21 = view_34.transpose(1, 2)
        view_34 = None
        query_rot_10 = query_states_31[(Ellipsis, slice(None, 20, None))]
        query_pass_10 = query_states_31[(Ellipsis, slice(20, None, None))]
        query_states_31 = None
        key_rot_10 = key_states_31[(Ellipsis, slice(None, 20, None))]
        key_pass_10 = key_states_31[(Ellipsis, slice(20, None, None))]
        key_states_31 = None
        getitem_126 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_30 = getitem_126.to(dtype=torch.float16)
        getitem_126 = None
        getitem_127 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_30 = getitem_127.to(dtype=torch.float16)
        getitem_127 = None
        squeeze_40 = cos_30.squeeze(1)
        cos_30 = None
        cos_31 = squeeze_40.squeeze(0)
        squeeze_40 = None
        squeeze_42 = sin_30.squeeze(1)
        sin_30 = None
        sin_31 = squeeze_42.squeeze(0)
        squeeze_42 = None
        getitem_128 = cos_31[position_ids_1]
        cos_31 = None
        cos_32 = getitem_128.unsqueeze(1)
        getitem_128 = None
        getitem_129 = sin_31[position_ids_1]
        sin_31 = None
        sin_32 = getitem_129.unsqueeze(1)
        getitem_129 = None
        mul_50 = query_rot_10 * cos_32
        chunk_20 = torch.chunk(query_rot_10, 2, dim=-1)
        query_rot_10 = None
        x1_20 = chunk_20[0]
        x2_20 = chunk_20[1]
        chunk_20 = None
        neg_20 = -x2_20
        x2_20 = None
        cat_40 = torch.cat((neg_20, x1_20), dim=-1)
        neg_20 = x1_20 = None
        mul_51 = cat_40 * sin_32
        cat_40 = None
        q_embed_10 = mul_50 + mul_51
        mul_50 = mul_51 = None
        mul_52 = key_rot_10 * cos_32
        cos_32 = None
        chunk_21 = torch.chunk(key_rot_10, 2, dim=-1)
        key_rot_10 = None
        x1_21 = chunk_21[0]
        x2_21 = chunk_21[1]
        chunk_21 = None
        neg_21 = -x2_21
        x2_21 = None
        cat_41 = torch.cat((neg_21, x1_21), dim=-1)
        neg_21 = x1_21 = None
        mul_53 = cat_41 * sin_32
        cat_41 = sin_32 = None
        k_embed_10 = mul_52 + mul_53
        mul_52 = mul_53 = None
        query_states_32 = torch.cat((q_embed_10, query_pass_10), dim=-1)
        q_embed_10 = query_pass_10 = None
        key_states_32 = torch.cat((k_embed_10, key_pass_10), dim=-1)
        k_embed_10 = key_pass_10 = None
        transpose_53 = key_states_32.transpose(2, 3)
        key_states_32 = None
        matmul_20 = torch.matmul(query_states_32, transpose_53)
        query_states_32 = transpose_53 = None
        attn_weights_30 = matmul_20 / 8.94427190999916
        matmul_20 = None
        attn_weights_31 = attn_weights_30 + combined_attention_mask_1
        attn_weights_30 = None
        softmax_10 = torch.nn.functional.softmax(
            attn_weights_31, dim=-1, dtype=torch.float32
        )
        attn_weights_31 = None
        attn_weights_32 = softmax_10.to(torch.float16)
        softmax_10 = None
        attn_output_40 = torch.matmul(attn_weights_32, value_states_21)
        attn_weights_32 = value_states_21 = None
        transpose_54 = attn_output_40.transpose(1, 2)
        attn_output_40 = None
        attn_output_41 = transpose_54.contiguous()
        transpose_54 = None
        attn_output_42 = attn_output_41.reshape(1, 19, 2560)
        attn_output_41 = None
        attn_output_43 = torch._C._nn.linear(
            attn_output_42,
            l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_42 = l_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_51 = hidden_states_49 + attn_output_43
        hidden_states_49 = attn_output_43 = None
        hidden_states_52 = torch.nn.functional.layer_norm(
            hidden_states_51,
            (2560,),
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_74 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_10 = torch.nn.functional.silu(linear_74, inplace=False)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            hidden_states_52,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_52 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_54 = silu_10 * linear_75
        silu_10 = linear_75 = None
        hidden_states_53 = torch._C._nn.linear(
            mul_54,
            l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_54 = l_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_54 = hidden_states_51 + hidden_states_53
        hidden_states_51 = hidden_states_53 = None
        hidden_states_55 = torch.nn.functional.layer_norm(
            hidden_states_54,
            (2560,),
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_bias_ = (None)
        query_states_33 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_33 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_22 = torch._C._nn.linear(
            hidden_states_55,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_55 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_35 = query_states_33.view(1, 19, 32, 80)
        query_states_33 = None
        query_states_34 = view_35.transpose(1, 2)
        view_35 = None
        view_36 = key_states_33.view(1, 19, 32, 80)
        key_states_33 = None
        key_states_34 = view_36.transpose(1, 2)
        view_36 = None
        view_37 = value_states_22.view(1, 19, 32, 80)
        value_states_22 = None
        value_states_23 = view_37.transpose(1, 2)
        view_37 = None
        query_rot_11 = query_states_34[(Ellipsis, slice(None, 20, None))]
        query_pass_11 = query_states_34[(Ellipsis, slice(20, None, None))]
        query_states_34 = None
        key_rot_11 = key_states_34[(Ellipsis, slice(None, 20, None))]
        key_pass_11 = key_states_34[(Ellipsis, slice(20, None, None))]
        key_states_34 = None
        getitem_138 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_33 = getitem_138.to(dtype=torch.float16)
        getitem_138 = None
        getitem_139 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_33 = getitem_139.to(dtype=torch.float16)
        getitem_139 = None
        squeeze_44 = cos_33.squeeze(1)
        cos_33 = None
        cos_34 = squeeze_44.squeeze(0)
        squeeze_44 = None
        squeeze_46 = sin_33.squeeze(1)
        sin_33 = None
        sin_34 = squeeze_46.squeeze(0)
        squeeze_46 = None
        getitem_140 = cos_34[position_ids_1]
        cos_34 = None
        cos_35 = getitem_140.unsqueeze(1)
        getitem_140 = None
        getitem_141 = sin_34[position_ids_1]
        sin_34 = None
        sin_35 = getitem_141.unsqueeze(1)
        getitem_141 = None
        mul_55 = query_rot_11 * cos_35
        chunk_22 = torch.chunk(query_rot_11, 2, dim=-1)
        query_rot_11 = None
        x1_22 = chunk_22[0]
        x2_22 = chunk_22[1]
        chunk_22 = None
        neg_22 = -x2_22
        x2_22 = None
        cat_44 = torch.cat((neg_22, x1_22), dim=-1)
        neg_22 = x1_22 = None
        mul_56 = cat_44 * sin_35
        cat_44 = None
        q_embed_11 = mul_55 + mul_56
        mul_55 = mul_56 = None
        mul_57 = key_rot_11 * cos_35
        cos_35 = None
        chunk_23 = torch.chunk(key_rot_11, 2, dim=-1)
        key_rot_11 = None
        x1_23 = chunk_23[0]
        x2_23 = chunk_23[1]
        chunk_23 = None
        neg_23 = -x2_23
        x2_23 = None
        cat_45 = torch.cat((neg_23, x1_23), dim=-1)
        neg_23 = x1_23 = None
        mul_58 = cat_45 * sin_35
        cat_45 = sin_35 = None
        k_embed_11 = mul_57 + mul_58
        mul_57 = mul_58 = None
        query_states_35 = torch.cat((q_embed_11, query_pass_11), dim=-1)
        q_embed_11 = query_pass_11 = None
        key_states_35 = torch.cat((k_embed_11, key_pass_11), dim=-1)
        k_embed_11 = key_pass_11 = None
        transpose_58 = key_states_35.transpose(2, 3)
        key_states_35 = None
        matmul_22 = torch.matmul(query_states_35, transpose_58)
        query_states_35 = transpose_58 = None
        attn_weights_33 = matmul_22 / 8.94427190999916
        matmul_22 = None
        attn_weights_34 = attn_weights_33 + combined_attention_mask_1
        attn_weights_33 = None
        softmax_11 = torch.nn.functional.softmax(
            attn_weights_34, dim=-1, dtype=torch.float32
        )
        attn_weights_34 = None
        attn_weights_35 = softmax_11.to(torch.float16)
        softmax_11 = None
        attn_output_44 = torch.matmul(attn_weights_35, value_states_23)
        attn_weights_35 = value_states_23 = None
        transpose_59 = attn_output_44.transpose(1, 2)
        attn_output_44 = None
        attn_output_45 = transpose_59.contiguous()
        transpose_59 = None
        attn_output_46 = attn_output_45.reshape(1, 19, 2560)
        attn_output_45 = None
        attn_output_47 = torch._C._nn.linear(
            attn_output_46,
            l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_46 = l_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_56 = hidden_states_54 + attn_output_47
        hidden_states_54 = attn_output_47 = None
        hidden_states_57 = torch.nn.functional.layer_norm(
            hidden_states_56,
            (2560,),
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_81 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_11 = torch.nn.functional.silu(linear_81, inplace=False)
        linear_81 = None
        linear_82 = torch._C._nn.linear(
            hidden_states_57,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_57 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_59 = silu_11 * linear_82
        silu_11 = linear_82 = None
        hidden_states_58 = torch._C._nn.linear(
            mul_59,
            l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_59 = l_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_59 = hidden_states_56 + hidden_states_58
        hidden_states_56 = hidden_states_58 = None
        hidden_states_60 = torch.nn.functional.layer_norm(
            hidden_states_59,
            (2560,),
            l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_bias_ = (None)
        query_states_36 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_36 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_24 = torch._C._nn.linear(
            hidden_states_60,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_60 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_38 = query_states_36.view(1, 19, 32, 80)
        query_states_36 = None
        query_states_37 = view_38.transpose(1, 2)
        view_38 = None
        view_39 = key_states_36.view(1, 19, 32, 80)
        key_states_36 = None
        key_states_37 = view_39.transpose(1, 2)
        view_39 = None
        view_40 = value_states_24.view(1, 19, 32, 80)
        value_states_24 = None
        value_states_25 = view_40.transpose(1, 2)
        view_40 = None
        query_rot_12 = query_states_37[(Ellipsis, slice(None, 20, None))]
        query_pass_12 = query_states_37[(Ellipsis, slice(20, None, None))]
        query_states_37 = None
        key_rot_12 = key_states_37[(Ellipsis, slice(None, 20, None))]
        key_pass_12 = key_states_37[(Ellipsis, slice(20, None, None))]
        key_states_37 = None
        getitem_150 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_36 = getitem_150.to(dtype=torch.float16)
        getitem_150 = None
        getitem_151 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_36 = getitem_151.to(dtype=torch.float16)
        getitem_151 = None
        squeeze_48 = cos_36.squeeze(1)
        cos_36 = None
        cos_37 = squeeze_48.squeeze(0)
        squeeze_48 = None
        squeeze_50 = sin_36.squeeze(1)
        sin_36 = None
        sin_37 = squeeze_50.squeeze(0)
        squeeze_50 = None
        getitem_152 = cos_37[position_ids_1]
        cos_37 = None
        cos_38 = getitem_152.unsqueeze(1)
        getitem_152 = None
        getitem_153 = sin_37[position_ids_1]
        sin_37 = None
        sin_38 = getitem_153.unsqueeze(1)
        getitem_153 = None
        mul_60 = query_rot_12 * cos_38
        chunk_24 = torch.chunk(query_rot_12, 2, dim=-1)
        query_rot_12 = None
        x1_24 = chunk_24[0]
        x2_24 = chunk_24[1]
        chunk_24 = None
        neg_24 = -x2_24
        x2_24 = None
        cat_48 = torch.cat((neg_24, x1_24), dim=-1)
        neg_24 = x1_24 = None
        mul_61 = cat_48 * sin_38
        cat_48 = None
        q_embed_12 = mul_60 + mul_61
        mul_60 = mul_61 = None
        mul_62 = key_rot_12 * cos_38
        cos_38 = None
        chunk_25 = torch.chunk(key_rot_12, 2, dim=-1)
        key_rot_12 = None
        x1_25 = chunk_25[0]
        x2_25 = chunk_25[1]
        chunk_25 = None
        neg_25 = -x2_25
        x2_25 = None
        cat_49 = torch.cat((neg_25, x1_25), dim=-1)
        neg_25 = x1_25 = None
        mul_63 = cat_49 * sin_38
        cat_49 = sin_38 = None
        k_embed_12 = mul_62 + mul_63
        mul_62 = mul_63 = None
        query_states_38 = torch.cat((q_embed_12, query_pass_12), dim=-1)
        q_embed_12 = query_pass_12 = None
        key_states_38 = torch.cat((k_embed_12, key_pass_12), dim=-1)
        k_embed_12 = key_pass_12 = None
        transpose_63 = key_states_38.transpose(2, 3)
        key_states_38 = None
        matmul_24 = torch.matmul(query_states_38, transpose_63)
        query_states_38 = transpose_63 = None
        attn_weights_36 = matmul_24 / 8.94427190999916
        matmul_24 = None
        attn_weights_37 = attn_weights_36 + combined_attention_mask_1
        attn_weights_36 = None
        softmax_12 = torch.nn.functional.softmax(
            attn_weights_37, dim=-1, dtype=torch.float32
        )
        attn_weights_37 = None
        attn_weights_38 = softmax_12.to(torch.float16)
        softmax_12 = None
        attn_output_48 = torch.matmul(attn_weights_38, value_states_25)
        attn_weights_38 = value_states_25 = None
        transpose_64 = attn_output_48.transpose(1, 2)
        attn_output_48 = None
        attn_output_49 = transpose_64.contiguous()
        transpose_64 = None
        attn_output_50 = attn_output_49.reshape(1, 19, 2560)
        attn_output_49 = None
        attn_output_51 = torch._C._nn.linear(
            attn_output_50,
            l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_50 = l_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_61 = hidden_states_59 + attn_output_51
        hidden_states_59 = attn_output_51 = None
        hidden_states_62 = torch.nn.functional.layer_norm(
            hidden_states_61,
            (2560,),
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_88 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_12 = torch.nn.functional.silu(linear_88, inplace=False)
        linear_88 = None
        linear_89 = torch._C._nn.linear(
            hidden_states_62,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_62 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_64 = silu_12 * linear_89
        silu_12 = linear_89 = None
        hidden_states_63 = torch._C._nn.linear(
            mul_64,
            l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_64 = l_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_64 = hidden_states_61 + hidden_states_63
        hidden_states_61 = hidden_states_63 = None
        hidden_states_65 = torch.nn.functional.layer_norm(
            hidden_states_64,
            (2560,),
            l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_bias_ = (None)
        query_states_39 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_39 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_26 = torch._C._nn.linear(
            hidden_states_65,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_65 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_41 = query_states_39.view(1, 19, 32, 80)
        query_states_39 = None
        query_states_40 = view_41.transpose(1, 2)
        view_41 = None
        view_42 = key_states_39.view(1, 19, 32, 80)
        key_states_39 = None
        key_states_40 = view_42.transpose(1, 2)
        view_42 = None
        view_43 = value_states_26.view(1, 19, 32, 80)
        value_states_26 = None
        value_states_27 = view_43.transpose(1, 2)
        view_43 = None
        query_rot_13 = query_states_40[(Ellipsis, slice(None, 20, None))]
        query_pass_13 = query_states_40[(Ellipsis, slice(20, None, None))]
        query_states_40 = None
        key_rot_13 = key_states_40[(Ellipsis, slice(None, 20, None))]
        key_pass_13 = key_states_40[(Ellipsis, slice(20, None, None))]
        key_states_40 = None
        getitem_162 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_39 = getitem_162.to(dtype=torch.float16)
        getitem_162 = None
        getitem_163 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_39 = getitem_163.to(dtype=torch.float16)
        getitem_163 = None
        squeeze_52 = cos_39.squeeze(1)
        cos_39 = None
        cos_40 = squeeze_52.squeeze(0)
        squeeze_52 = None
        squeeze_54 = sin_39.squeeze(1)
        sin_39 = None
        sin_40 = squeeze_54.squeeze(0)
        squeeze_54 = None
        getitem_164 = cos_40[position_ids_1]
        cos_40 = None
        cos_41 = getitem_164.unsqueeze(1)
        getitem_164 = None
        getitem_165 = sin_40[position_ids_1]
        sin_40 = None
        sin_41 = getitem_165.unsqueeze(1)
        getitem_165 = None
        mul_65 = query_rot_13 * cos_41
        chunk_26 = torch.chunk(query_rot_13, 2, dim=-1)
        query_rot_13 = None
        x1_26 = chunk_26[0]
        x2_26 = chunk_26[1]
        chunk_26 = None
        neg_26 = -x2_26
        x2_26 = None
        cat_52 = torch.cat((neg_26, x1_26), dim=-1)
        neg_26 = x1_26 = None
        mul_66 = cat_52 * sin_41
        cat_52 = None
        q_embed_13 = mul_65 + mul_66
        mul_65 = mul_66 = None
        mul_67 = key_rot_13 * cos_41
        cos_41 = None
        chunk_27 = torch.chunk(key_rot_13, 2, dim=-1)
        key_rot_13 = None
        x1_27 = chunk_27[0]
        x2_27 = chunk_27[1]
        chunk_27 = None
        neg_27 = -x2_27
        x2_27 = None
        cat_53 = torch.cat((neg_27, x1_27), dim=-1)
        neg_27 = x1_27 = None
        mul_68 = cat_53 * sin_41
        cat_53 = sin_41 = None
        k_embed_13 = mul_67 + mul_68
        mul_67 = mul_68 = None
        query_states_41 = torch.cat((q_embed_13, query_pass_13), dim=-1)
        q_embed_13 = query_pass_13 = None
        key_states_41 = torch.cat((k_embed_13, key_pass_13), dim=-1)
        k_embed_13 = key_pass_13 = None
        transpose_68 = key_states_41.transpose(2, 3)
        key_states_41 = None
        matmul_26 = torch.matmul(query_states_41, transpose_68)
        query_states_41 = transpose_68 = None
        attn_weights_39 = matmul_26 / 8.94427190999916
        matmul_26 = None
        attn_weights_40 = attn_weights_39 + combined_attention_mask_1
        attn_weights_39 = None
        softmax_13 = torch.nn.functional.softmax(
            attn_weights_40, dim=-1, dtype=torch.float32
        )
        attn_weights_40 = None
        attn_weights_41 = softmax_13.to(torch.float16)
        softmax_13 = None
        attn_output_52 = torch.matmul(attn_weights_41, value_states_27)
        attn_weights_41 = value_states_27 = None
        transpose_69 = attn_output_52.transpose(1, 2)
        attn_output_52 = None
        attn_output_53 = transpose_69.contiguous()
        transpose_69 = None
        attn_output_54 = attn_output_53.reshape(1, 19, 2560)
        attn_output_53 = None
        attn_output_55 = torch._C._nn.linear(
            attn_output_54,
            l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_54 = l_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_66 = hidden_states_64 + attn_output_55
        hidden_states_64 = attn_output_55 = None
        hidden_states_67 = torch.nn.functional.layer_norm(
            hidden_states_66,
            (2560,),
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_95 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_13 = torch.nn.functional.silu(linear_95, inplace=False)
        linear_95 = None
        linear_96 = torch._C._nn.linear(
            hidden_states_67,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_67 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_69 = silu_13 * linear_96
        silu_13 = linear_96 = None
        hidden_states_68 = torch._C._nn.linear(
            mul_69,
            l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_69 = l_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_69 = hidden_states_66 + hidden_states_68
        hidden_states_66 = hidden_states_68 = None
        hidden_states_70 = torch.nn.functional.layer_norm(
            hidden_states_69,
            (2560,),
            l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_bias_ = (None)
        query_states_42 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_42 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_28 = torch._C._nn.linear(
            hidden_states_70,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_70 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_44 = query_states_42.view(1, 19, 32, 80)
        query_states_42 = None
        query_states_43 = view_44.transpose(1, 2)
        view_44 = None
        view_45 = key_states_42.view(1, 19, 32, 80)
        key_states_42 = None
        key_states_43 = view_45.transpose(1, 2)
        view_45 = None
        view_46 = value_states_28.view(1, 19, 32, 80)
        value_states_28 = None
        value_states_29 = view_46.transpose(1, 2)
        view_46 = None
        query_rot_14 = query_states_43[(Ellipsis, slice(None, 20, None))]
        query_pass_14 = query_states_43[(Ellipsis, slice(20, None, None))]
        query_states_43 = None
        key_rot_14 = key_states_43[(Ellipsis, slice(None, 20, None))]
        key_pass_14 = key_states_43[(Ellipsis, slice(20, None, None))]
        key_states_43 = None
        getitem_174 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_42 = getitem_174.to(dtype=torch.float16)
        getitem_174 = None
        getitem_175 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_42 = getitem_175.to(dtype=torch.float16)
        getitem_175 = None
        squeeze_56 = cos_42.squeeze(1)
        cos_42 = None
        cos_43 = squeeze_56.squeeze(0)
        squeeze_56 = None
        squeeze_58 = sin_42.squeeze(1)
        sin_42 = None
        sin_43 = squeeze_58.squeeze(0)
        squeeze_58 = None
        getitem_176 = cos_43[position_ids_1]
        cos_43 = None
        cos_44 = getitem_176.unsqueeze(1)
        getitem_176 = None
        getitem_177 = sin_43[position_ids_1]
        sin_43 = None
        sin_44 = getitem_177.unsqueeze(1)
        getitem_177 = None
        mul_70 = query_rot_14 * cos_44
        chunk_28 = torch.chunk(query_rot_14, 2, dim=-1)
        query_rot_14 = None
        x1_28 = chunk_28[0]
        x2_28 = chunk_28[1]
        chunk_28 = None
        neg_28 = -x2_28
        x2_28 = None
        cat_56 = torch.cat((neg_28, x1_28), dim=-1)
        neg_28 = x1_28 = None
        mul_71 = cat_56 * sin_44
        cat_56 = None
        q_embed_14 = mul_70 + mul_71
        mul_70 = mul_71 = None
        mul_72 = key_rot_14 * cos_44
        cos_44 = None
        chunk_29 = torch.chunk(key_rot_14, 2, dim=-1)
        key_rot_14 = None
        x1_29 = chunk_29[0]
        x2_29 = chunk_29[1]
        chunk_29 = None
        neg_29 = -x2_29
        x2_29 = None
        cat_57 = torch.cat((neg_29, x1_29), dim=-1)
        neg_29 = x1_29 = None
        mul_73 = cat_57 * sin_44
        cat_57 = sin_44 = None
        k_embed_14 = mul_72 + mul_73
        mul_72 = mul_73 = None
        query_states_44 = torch.cat((q_embed_14, query_pass_14), dim=-1)
        q_embed_14 = query_pass_14 = None
        key_states_44 = torch.cat((k_embed_14, key_pass_14), dim=-1)
        k_embed_14 = key_pass_14 = None
        transpose_73 = key_states_44.transpose(2, 3)
        key_states_44 = None
        matmul_28 = torch.matmul(query_states_44, transpose_73)
        query_states_44 = transpose_73 = None
        attn_weights_42 = matmul_28 / 8.94427190999916
        matmul_28 = None
        attn_weights_43 = attn_weights_42 + combined_attention_mask_1
        attn_weights_42 = None
        softmax_14 = torch.nn.functional.softmax(
            attn_weights_43, dim=-1, dtype=torch.float32
        )
        attn_weights_43 = None
        attn_weights_44 = softmax_14.to(torch.float16)
        softmax_14 = None
        attn_output_56 = torch.matmul(attn_weights_44, value_states_29)
        attn_weights_44 = value_states_29 = None
        transpose_74 = attn_output_56.transpose(1, 2)
        attn_output_56 = None
        attn_output_57 = transpose_74.contiguous()
        transpose_74 = None
        attn_output_58 = attn_output_57.reshape(1, 19, 2560)
        attn_output_57 = None
        attn_output_59 = torch._C._nn.linear(
            attn_output_58,
            l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_58 = l_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_71 = hidden_states_69 + attn_output_59
        hidden_states_69 = attn_output_59 = None
        hidden_states_72 = torch.nn.functional.layer_norm(
            hidden_states_71,
            (2560,),
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_102 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_14 = torch.nn.functional.silu(linear_102, inplace=False)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            hidden_states_72,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_72 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_74 = silu_14 * linear_103
        silu_14 = linear_103 = None
        hidden_states_73 = torch._C._nn.linear(
            mul_74,
            l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_74 = l_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_74 = hidden_states_71 + hidden_states_73
        hidden_states_71 = hidden_states_73 = None
        hidden_states_75 = torch.nn.functional.layer_norm(
            hidden_states_74,
            (2560,),
            l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_bias_ = (None)
        query_states_45 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_45 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_30 = torch._C._nn.linear(
            hidden_states_75,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_75 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_47 = query_states_45.view(1, 19, 32, 80)
        query_states_45 = None
        query_states_46 = view_47.transpose(1, 2)
        view_47 = None
        view_48 = key_states_45.view(1, 19, 32, 80)
        key_states_45 = None
        key_states_46 = view_48.transpose(1, 2)
        view_48 = None
        view_49 = value_states_30.view(1, 19, 32, 80)
        value_states_30 = None
        value_states_31 = view_49.transpose(1, 2)
        view_49 = None
        query_rot_15 = query_states_46[(Ellipsis, slice(None, 20, None))]
        query_pass_15 = query_states_46[(Ellipsis, slice(20, None, None))]
        query_states_46 = None
        key_rot_15 = key_states_46[(Ellipsis, slice(None, 20, None))]
        key_pass_15 = key_states_46[(Ellipsis, slice(20, None, None))]
        key_states_46 = None
        getitem_186 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_45 = getitem_186.to(dtype=torch.float16)
        getitem_186 = None
        getitem_187 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_45 = getitem_187.to(dtype=torch.float16)
        getitem_187 = None
        squeeze_60 = cos_45.squeeze(1)
        cos_45 = None
        cos_46 = squeeze_60.squeeze(0)
        squeeze_60 = None
        squeeze_62 = sin_45.squeeze(1)
        sin_45 = None
        sin_46 = squeeze_62.squeeze(0)
        squeeze_62 = None
        getitem_188 = cos_46[position_ids_1]
        cos_46 = None
        cos_47 = getitem_188.unsqueeze(1)
        getitem_188 = None
        getitem_189 = sin_46[position_ids_1]
        sin_46 = None
        sin_47 = getitem_189.unsqueeze(1)
        getitem_189 = None
        mul_75 = query_rot_15 * cos_47
        chunk_30 = torch.chunk(query_rot_15, 2, dim=-1)
        query_rot_15 = None
        x1_30 = chunk_30[0]
        x2_30 = chunk_30[1]
        chunk_30 = None
        neg_30 = -x2_30
        x2_30 = None
        cat_60 = torch.cat((neg_30, x1_30), dim=-1)
        neg_30 = x1_30 = None
        mul_76 = cat_60 * sin_47
        cat_60 = None
        q_embed_15 = mul_75 + mul_76
        mul_75 = mul_76 = None
        mul_77 = key_rot_15 * cos_47
        cos_47 = None
        chunk_31 = torch.chunk(key_rot_15, 2, dim=-1)
        key_rot_15 = None
        x1_31 = chunk_31[0]
        x2_31 = chunk_31[1]
        chunk_31 = None
        neg_31 = -x2_31
        x2_31 = None
        cat_61 = torch.cat((neg_31, x1_31), dim=-1)
        neg_31 = x1_31 = None
        mul_78 = cat_61 * sin_47
        cat_61 = sin_47 = None
        k_embed_15 = mul_77 + mul_78
        mul_77 = mul_78 = None
        query_states_47 = torch.cat((q_embed_15, query_pass_15), dim=-1)
        q_embed_15 = query_pass_15 = None
        key_states_47 = torch.cat((k_embed_15, key_pass_15), dim=-1)
        k_embed_15 = key_pass_15 = None
        transpose_78 = key_states_47.transpose(2, 3)
        key_states_47 = None
        matmul_30 = torch.matmul(query_states_47, transpose_78)
        query_states_47 = transpose_78 = None
        attn_weights_45 = matmul_30 / 8.94427190999916
        matmul_30 = None
        attn_weights_46 = attn_weights_45 + combined_attention_mask_1
        attn_weights_45 = None
        softmax_15 = torch.nn.functional.softmax(
            attn_weights_46, dim=-1, dtype=torch.float32
        )
        attn_weights_46 = None
        attn_weights_47 = softmax_15.to(torch.float16)
        softmax_15 = None
        attn_output_60 = torch.matmul(attn_weights_47, value_states_31)
        attn_weights_47 = value_states_31 = None
        transpose_79 = attn_output_60.transpose(1, 2)
        attn_output_60 = None
        attn_output_61 = transpose_79.contiguous()
        transpose_79 = None
        attn_output_62 = attn_output_61.reshape(1, 19, 2560)
        attn_output_61 = None
        attn_output_63 = torch._C._nn.linear(
            attn_output_62,
            l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_62 = l_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_76 = hidden_states_74 + attn_output_63
        hidden_states_74 = attn_output_63 = None
        hidden_states_77 = torch.nn.functional.layer_norm(
            hidden_states_76,
            (2560,),
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_109 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_15 = torch.nn.functional.silu(linear_109, inplace=False)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            hidden_states_77,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_77 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_79 = silu_15 * linear_110
        silu_15 = linear_110 = None
        hidden_states_78 = torch._C._nn.linear(
            mul_79,
            l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_79 = l_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_79 = hidden_states_76 + hidden_states_78
        hidden_states_76 = hidden_states_78 = None
        hidden_states_80 = torch.nn.functional.layer_norm(
            hidden_states_79,
            (2560,),
            l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_bias_ = (None)
        query_states_48 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_48 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_32 = torch._C._nn.linear(
            hidden_states_80,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_80 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_50 = query_states_48.view(1, 19, 32, 80)
        query_states_48 = None
        query_states_49 = view_50.transpose(1, 2)
        view_50 = None
        view_51 = key_states_48.view(1, 19, 32, 80)
        key_states_48 = None
        key_states_49 = view_51.transpose(1, 2)
        view_51 = None
        view_52 = value_states_32.view(1, 19, 32, 80)
        value_states_32 = None
        value_states_33 = view_52.transpose(1, 2)
        view_52 = None
        query_rot_16 = query_states_49[(Ellipsis, slice(None, 20, None))]
        query_pass_16 = query_states_49[(Ellipsis, slice(20, None, None))]
        query_states_49 = None
        key_rot_16 = key_states_49[(Ellipsis, slice(None, 20, None))]
        key_pass_16 = key_states_49[(Ellipsis, slice(20, None, None))]
        key_states_49 = None
        getitem_198 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_48 = getitem_198.to(dtype=torch.float16)
        getitem_198 = None
        getitem_199 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_48 = getitem_199.to(dtype=torch.float16)
        getitem_199 = None
        squeeze_64 = cos_48.squeeze(1)
        cos_48 = None
        cos_49 = squeeze_64.squeeze(0)
        squeeze_64 = None
        squeeze_66 = sin_48.squeeze(1)
        sin_48 = None
        sin_49 = squeeze_66.squeeze(0)
        squeeze_66 = None
        getitem_200 = cos_49[position_ids_1]
        cos_49 = None
        cos_50 = getitem_200.unsqueeze(1)
        getitem_200 = None
        getitem_201 = sin_49[position_ids_1]
        sin_49 = None
        sin_50 = getitem_201.unsqueeze(1)
        getitem_201 = None
        mul_80 = query_rot_16 * cos_50
        chunk_32 = torch.chunk(query_rot_16, 2, dim=-1)
        query_rot_16 = None
        x1_32 = chunk_32[0]
        x2_32 = chunk_32[1]
        chunk_32 = None
        neg_32 = -x2_32
        x2_32 = None
        cat_64 = torch.cat((neg_32, x1_32), dim=-1)
        neg_32 = x1_32 = None
        mul_81 = cat_64 * sin_50
        cat_64 = None
        q_embed_16 = mul_80 + mul_81
        mul_80 = mul_81 = None
        mul_82 = key_rot_16 * cos_50
        cos_50 = None
        chunk_33 = torch.chunk(key_rot_16, 2, dim=-1)
        key_rot_16 = None
        x1_33 = chunk_33[0]
        x2_33 = chunk_33[1]
        chunk_33 = None
        neg_33 = -x2_33
        x2_33 = None
        cat_65 = torch.cat((neg_33, x1_33), dim=-1)
        neg_33 = x1_33 = None
        mul_83 = cat_65 * sin_50
        cat_65 = sin_50 = None
        k_embed_16 = mul_82 + mul_83
        mul_82 = mul_83 = None
        query_states_50 = torch.cat((q_embed_16, query_pass_16), dim=-1)
        q_embed_16 = query_pass_16 = None
        key_states_50 = torch.cat((k_embed_16, key_pass_16), dim=-1)
        k_embed_16 = key_pass_16 = None
        transpose_83 = key_states_50.transpose(2, 3)
        key_states_50 = None
        matmul_32 = torch.matmul(query_states_50, transpose_83)
        query_states_50 = transpose_83 = None
        attn_weights_48 = matmul_32 / 8.94427190999916
        matmul_32 = None
        attn_weights_49 = attn_weights_48 + combined_attention_mask_1
        attn_weights_48 = None
        softmax_16 = torch.nn.functional.softmax(
            attn_weights_49, dim=-1, dtype=torch.float32
        )
        attn_weights_49 = None
        attn_weights_50 = softmax_16.to(torch.float16)
        softmax_16 = None
        attn_output_64 = torch.matmul(attn_weights_50, value_states_33)
        attn_weights_50 = value_states_33 = None
        transpose_84 = attn_output_64.transpose(1, 2)
        attn_output_64 = None
        attn_output_65 = transpose_84.contiguous()
        transpose_84 = None
        attn_output_66 = attn_output_65.reshape(1, 19, 2560)
        attn_output_65 = None
        attn_output_67 = torch._C._nn.linear(
            attn_output_66,
            l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_66 = l_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_81 = hidden_states_79 + attn_output_67
        hidden_states_79 = attn_output_67 = None
        hidden_states_82 = torch.nn.functional.layer_norm(
            hidden_states_81,
            (2560,),
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_116 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_16 = torch.nn.functional.silu(linear_116, inplace=False)
        linear_116 = None
        linear_117 = torch._C._nn.linear(
            hidden_states_82,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_82 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_84 = silu_16 * linear_117
        silu_16 = linear_117 = None
        hidden_states_83 = torch._C._nn.linear(
            mul_84,
            l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_84 = l_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_84 = hidden_states_81 + hidden_states_83
        hidden_states_81 = hidden_states_83 = None
        hidden_states_85 = torch.nn.functional.layer_norm(
            hidden_states_84,
            (2560,),
            l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_bias_ = (None)
        query_states_51 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_51 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_34 = torch._C._nn.linear(
            hidden_states_85,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_85 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_53 = query_states_51.view(1, 19, 32, 80)
        query_states_51 = None
        query_states_52 = view_53.transpose(1, 2)
        view_53 = None
        view_54 = key_states_51.view(1, 19, 32, 80)
        key_states_51 = None
        key_states_52 = view_54.transpose(1, 2)
        view_54 = None
        view_55 = value_states_34.view(1, 19, 32, 80)
        value_states_34 = None
        value_states_35 = view_55.transpose(1, 2)
        view_55 = None
        query_rot_17 = query_states_52[(Ellipsis, slice(None, 20, None))]
        query_pass_17 = query_states_52[(Ellipsis, slice(20, None, None))]
        query_states_52 = None
        key_rot_17 = key_states_52[(Ellipsis, slice(None, 20, None))]
        key_pass_17 = key_states_52[(Ellipsis, slice(20, None, None))]
        key_states_52 = None
        getitem_210 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_51 = getitem_210.to(dtype=torch.float16)
        getitem_210 = None
        getitem_211 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_51 = getitem_211.to(dtype=torch.float16)
        getitem_211 = None
        squeeze_68 = cos_51.squeeze(1)
        cos_51 = None
        cos_52 = squeeze_68.squeeze(0)
        squeeze_68 = None
        squeeze_70 = sin_51.squeeze(1)
        sin_51 = None
        sin_52 = squeeze_70.squeeze(0)
        squeeze_70 = None
        getitem_212 = cos_52[position_ids_1]
        cos_52 = None
        cos_53 = getitem_212.unsqueeze(1)
        getitem_212 = None
        getitem_213 = sin_52[position_ids_1]
        sin_52 = None
        sin_53 = getitem_213.unsqueeze(1)
        getitem_213 = None
        mul_85 = query_rot_17 * cos_53
        chunk_34 = torch.chunk(query_rot_17, 2, dim=-1)
        query_rot_17 = None
        x1_34 = chunk_34[0]
        x2_34 = chunk_34[1]
        chunk_34 = None
        neg_34 = -x2_34
        x2_34 = None
        cat_68 = torch.cat((neg_34, x1_34), dim=-1)
        neg_34 = x1_34 = None
        mul_86 = cat_68 * sin_53
        cat_68 = None
        q_embed_17 = mul_85 + mul_86
        mul_85 = mul_86 = None
        mul_87 = key_rot_17 * cos_53
        cos_53 = None
        chunk_35 = torch.chunk(key_rot_17, 2, dim=-1)
        key_rot_17 = None
        x1_35 = chunk_35[0]
        x2_35 = chunk_35[1]
        chunk_35 = None
        neg_35 = -x2_35
        x2_35 = None
        cat_69 = torch.cat((neg_35, x1_35), dim=-1)
        neg_35 = x1_35 = None
        mul_88 = cat_69 * sin_53
        cat_69 = sin_53 = None
        k_embed_17 = mul_87 + mul_88
        mul_87 = mul_88 = None
        query_states_53 = torch.cat((q_embed_17, query_pass_17), dim=-1)
        q_embed_17 = query_pass_17 = None
        key_states_53 = torch.cat((k_embed_17, key_pass_17), dim=-1)
        k_embed_17 = key_pass_17 = None
        transpose_88 = key_states_53.transpose(2, 3)
        key_states_53 = None
        matmul_34 = torch.matmul(query_states_53, transpose_88)
        query_states_53 = transpose_88 = None
        attn_weights_51 = matmul_34 / 8.94427190999916
        matmul_34 = None
        attn_weights_52 = attn_weights_51 + combined_attention_mask_1
        attn_weights_51 = None
        softmax_17 = torch.nn.functional.softmax(
            attn_weights_52, dim=-1, dtype=torch.float32
        )
        attn_weights_52 = None
        attn_weights_53 = softmax_17.to(torch.float16)
        softmax_17 = None
        attn_output_68 = torch.matmul(attn_weights_53, value_states_35)
        attn_weights_53 = value_states_35 = None
        transpose_89 = attn_output_68.transpose(1, 2)
        attn_output_68 = None
        attn_output_69 = transpose_89.contiguous()
        transpose_89 = None
        attn_output_70 = attn_output_69.reshape(1, 19, 2560)
        attn_output_69 = None
        attn_output_71 = torch._C._nn.linear(
            attn_output_70,
            l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_70 = l_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_86 = hidden_states_84 + attn_output_71
        hidden_states_84 = attn_output_71 = None
        hidden_states_87 = torch.nn.functional.layer_norm(
            hidden_states_86,
            (2560,),
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_123 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_17 = torch.nn.functional.silu(linear_123, inplace=False)
        linear_123 = None
        linear_124 = torch._C._nn.linear(
            hidden_states_87,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_87 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_89 = silu_17 * linear_124
        silu_17 = linear_124 = None
        hidden_states_88 = torch._C._nn.linear(
            mul_89,
            l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_89 = l_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_89 = hidden_states_86 + hidden_states_88
        hidden_states_86 = hidden_states_88 = None
        hidden_states_90 = torch.nn.functional.layer_norm(
            hidden_states_89,
            (2560,),
            l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_bias_ = (None)
        query_states_54 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_54 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_36 = torch._C._nn.linear(
            hidden_states_90,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_90 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_56 = query_states_54.view(1, 19, 32, 80)
        query_states_54 = None
        query_states_55 = view_56.transpose(1, 2)
        view_56 = None
        view_57 = key_states_54.view(1, 19, 32, 80)
        key_states_54 = None
        key_states_55 = view_57.transpose(1, 2)
        view_57 = None
        view_58 = value_states_36.view(1, 19, 32, 80)
        value_states_36 = None
        value_states_37 = view_58.transpose(1, 2)
        view_58 = None
        query_rot_18 = query_states_55[(Ellipsis, slice(None, 20, None))]
        query_pass_18 = query_states_55[(Ellipsis, slice(20, None, None))]
        query_states_55 = None
        key_rot_18 = key_states_55[(Ellipsis, slice(None, 20, None))]
        key_pass_18 = key_states_55[(Ellipsis, slice(20, None, None))]
        key_states_55 = None
        getitem_222 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_54 = getitem_222.to(dtype=torch.float16)
        getitem_222 = None
        getitem_223 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_54 = getitem_223.to(dtype=torch.float16)
        getitem_223 = None
        squeeze_72 = cos_54.squeeze(1)
        cos_54 = None
        cos_55 = squeeze_72.squeeze(0)
        squeeze_72 = None
        squeeze_74 = sin_54.squeeze(1)
        sin_54 = None
        sin_55 = squeeze_74.squeeze(0)
        squeeze_74 = None
        getitem_224 = cos_55[position_ids_1]
        cos_55 = None
        cos_56 = getitem_224.unsqueeze(1)
        getitem_224 = None
        getitem_225 = sin_55[position_ids_1]
        sin_55 = None
        sin_56 = getitem_225.unsqueeze(1)
        getitem_225 = None
        mul_90 = query_rot_18 * cos_56
        chunk_36 = torch.chunk(query_rot_18, 2, dim=-1)
        query_rot_18 = None
        x1_36 = chunk_36[0]
        x2_36 = chunk_36[1]
        chunk_36 = None
        neg_36 = -x2_36
        x2_36 = None
        cat_72 = torch.cat((neg_36, x1_36), dim=-1)
        neg_36 = x1_36 = None
        mul_91 = cat_72 * sin_56
        cat_72 = None
        q_embed_18 = mul_90 + mul_91
        mul_90 = mul_91 = None
        mul_92 = key_rot_18 * cos_56
        cos_56 = None
        chunk_37 = torch.chunk(key_rot_18, 2, dim=-1)
        key_rot_18 = None
        x1_37 = chunk_37[0]
        x2_37 = chunk_37[1]
        chunk_37 = None
        neg_37 = -x2_37
        x2_37 = None
        cat_73 = torch.cat((neg_37, x1_37), dim=-1)
        neg_37 = x1_37 = None
        mul_93 = cat_73 * sin_56
        cat_73 = sin_56 = None
        k_embed_18 = mul_92 + mul_93
        mul_92 = mul_93 = None
        query_states_56 = torch.cat((q_embed_18, query_pass_18), dim=-1)
        q_embed_18 = query_pass_18 = None
        key_states_56 = torch.cat((k_embed_18, key_pass_18), dim=-1)
        k_embed_18 = key_pass_18 = None
        transpose_93 = key_states_56.transpose(2, 3)
        key_states_56 = None
        matmul_36 = torch.matmul(query_states_56, transpose_93)
        query_states_56 = transpose_93 = None
        attn_weights_54 = matmul_36 / 8.94427190999916
        matmul_36 = None
        attn_weights_55 = attn_weights_54 + combined_attention_mask_1
        attn_weights_54 = None
        softmax_18 = torch.nn.functional.softmax(
            attn_weights_55, dim=-1, dtype=torch.float32
        )
        attn_weights_55 = None
        attn_weights_56 = softmax_18.to(torch.float16)
        softmax_18 = None
        attn_output_72 = torch.matmul(attn_weights_56, value_states_37)
        attn_weights_56 = value_states_37 = None
        transpose_94 = attn_output_72.transpose(1, 2)
        attn_output_72 = None
        attn_output_73 = transpose_94.contiguous()
        transpose_94 = None
        attn_output_74 = attn_output_73.reshape(1, 19, 2560)
        attn_output_73 = None
        attn_output_75 = torch._C._nn.linear(
            attn_output_74,
            l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_74 = l_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_91 = hidden_states_89 + attn_output_75
        hidden_states_89 = attn_output_75 = None
        hidden_states_92 = torch.nn.functional.layer_norm(
            hidden_states_91,
            (2560,),
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_130 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_18 = torch.nn.functional.silu(linear_130, inplace=False)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            hidden_states_92,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_92 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_94 = silu_18 * linear_131
        silu_18 = linear_131 = None
        hidden_states_93 = torch._C._nn.linear(
            mul_94,
            l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_94 = l_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_94 = hidden_states_91 + hidden_states_93
        hidden_states_91 = hidden_states_93 = None
        hidden_states_95 = torch.nn.functional.layer_norm(
            hidden_states_94,
            (2560,),
            l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_bias_ = (None)
        query_states_57 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_57 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_38 = torch._C._nn.linear(
            hidden_states_95,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_95 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_59 = query_states_57.view(1, 19, 32, 80)
        query_states_57 = None
        query_states_58 = view_59.transpose(1, 2)
        view_59 = None
        view_60 = key_states_57.view(1, 19, 32, 80)
        key_states_57 = None
        key_states_58 = view_60.transpose(1, 2)
        view_60 = None
        view_61 = value_states_38.view(1, 19, 32, 80)
        value_states_38 = None
        value_states_39 = view_61.transpose(1, 2)
        view_61 = None
        query_rot_19 = query_states_58[(Ellipsis, slice(None, 20, None))]
        query_pass_19 = query_states_58[(Ellipsis, slice(20, None, None))]
        query_states_58 = None
        key_rot_19 = key_states_58[(Ellipsis, slice(None, 20, None))]
        key_pass_19 = key_states_58[(Ellipsis, slice(20, None, None))]
        key_states_58 = None
        getitem_234 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_57 = getitem_234.to(dtype=torch.float16)
        getitem_234 = None
        getitem_235 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_57 = getitem_235.to(dtype=torch.float16)
        getitem_235 = None
        squeeze_76 = cos_57.squeeze(1)
        cos_57 = None
        cos_58 = squeeze_76.squeeze(0)
        squeeze_76 = None
        squeeze_78 = sin_57.squeeze(1)
        sin_57 = None
        sin_58 = squeeze_78.squeeze(0)
        squeeze_78 = None
        getitem_236 = cos_58[position_ids_1]
        cos_58 = None
        cos_59 = getitem_236.unsqueeze(1)
        getitem_236 = None
        getitem_237 = sin_58[position_ids_1]
        sin_58 = None
        sin_59 = getitem_237.unsqueeze(1)
        getitem_237 = None
        mul_95 = query_rot_19 * cos_59
        chunk_38 = torch.chunk(query_rot_19, 2, dim=-1)
        query_rot_19 = None
        x1_38 = chunk_38[0]
        x2_38 = chunk_38[1]
        chunk_38 = None
        neg_38 = -x2_38
        x2_38 = None
        cat_76 = torch.cat((neg_38, x1_38), dim=-1)
        neg_38 = x1_38 = None
        mul_96 = cat_76 * sin_59
        cat_76 = None
        q_embed_19 = mul_95 + mul_96
        mul_95 = mul_96 = None
        mul_97 = key_rot_19 * cos_59
        cos_59 = None
        chunk_39 = torch.chunk(key_rot_19, 2, dim=-1)
        key_rot_19 = None
        x1_39 = chunk_39[0]
        x2_39 = chunk_39[1]
        chunk_39 = None
        neg_39 = -x2_39
        x2_39 = None
        cat_77 = torch.cat((neg_39, x1_39), dim=-1)
        neg_39 = x1_39 = None
        mul_98 = cat_77 * sin_59
        cat_77 = sin_59 = None
        k_embed_19 = mul_97 + mul_98
        mul_97 = mul_98 = None
        query_states_59 = torch.cat((q_embed_19, query_pass_19), dim=-1)
        q_embed_19 = query_pass_19 = None
        key_states_59 = torch.cat((k_embed_19, key_pass_19), dim=-1)
        k_embed_19 = key_pass_19 = None
        transpose_98 = key_states_59.transpose(2, 3)
        key_states_59 = None
        matmul_38 = torch.matmul(query_states_59, transpose_98)
        query_states_59 = transpose_98 = None
        attn_weights_57 = matmul_38 / 8.94427190999916
        matmul_38 = None
        attn_weights_58 = attn_weights_57 + combined_attention_mask_1
        attn_weights_57 = None
        softmax_19 = torch.nn.functional.softmax(
            attn_weights_58, dim=-1, dtype=torch.float32
        )
        attn_weights_58 = None
        attn_weights_59 = softmax_19.to(torch.float16)
        softmax_19 = None
        attn_output_76 = torch.matmul(attn_weights_59, value_states_39)
        attn_weights_59 = value_states_39 = None
        transpose_99 = attn_output_76.transpose(1, 2)
        attn_output_76 = None
        attn_output_77 = transpose_99.contiguous()
        transpose_99 = None
        attn_output_78 = attn_output_77.reshape(1, 19, 2560)
        attn_output_77 = None
        attn_output_79 = torch._C._nn.linear(
            attn_output_78,
            l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_78 = l_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_96 = hidden_states_94 + attn_output_79
        hidden_states_94 = attn_output_79 = None
        hidden_states_97 = torch.nn.functional.layer_norm(
            hidden_states_96,
            (2560,),
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_137 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_19 = torch.nn.functional.silu(linear_137, inplace=False)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            hidden_states_97,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_97 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_99 = silu_19 * linear_138
        silu_19 = linear_138 = None
        hidden_states_98 = torch._C._nn.linear(
            mul_99,
            l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_99 = l_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_99 = hidden_states_96 + hidden_states_98
        hidden_states_96 = hidden_states_98 = None
        hidden_states_100 = torch.nn.functional.layer_norm(
            hidden_states_99,
            (2560,),
            l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_bias_ = (None)
        query_states_60 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_60 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_40 = torch._C._nn.linear(
            hidden_states_100,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_100 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_62 = query_states_60.view(1, 19, 32, 80)
        query_states_60 = None
        query_states_61 = view_62.transpose(1, 2)
        view_62 = None
        view_63 = key_states_60.view(1, 19, 32, 80)
        key_states_60 = None
        key_states_61 = view_63.transpose(1, 2)
        view_63 = None
        view_64 = value_states_40.view(1, 19, 32, 80)
        value_states_40 = None
        value_states_41 = view_64.transpose(1, 2)
        view_64 = None
        query_rot_20 = query_states_61[(Ellipsis, slice(None, 20, None))]
        query_pass_20 = query_states_61[(Ellipsis, slice(20, None, None))]
        query_states_61 = None
        key_rot_20 = key_states_61[(Ellipsis, slice(None, 20, None))]
        key_pass_20 = key_states_61[(Ellipsis, slice(20, None, None))]
        key_states_61 = None
        getitem_246 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_60 = getitem_246.to(dtype=torch.float16)
        getitem_246 = None
        getitem_247 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_60 = getitem_247.to(dtype=torch.float16)
        getitem_247 = None
        squeeze_80 = cos_60.squeeze(1)
        cos_60 = None
        cos_61 = squeeze_80.squeeze(0)
        squeeze_80 = None
        squeeze_82 = sin_60.squeeze(1)
        sin_60 = None
        sin_61 = squeeze_82.squeeze(0)
        squeeze_82 = None
        getitem_248 = cos_61[position_ids_1]
        cos_61 = None
        cos_62 = getitem_248.unsqueeze(1)
        getitem_248 = None
        getitem_249 = sin_61[position_ids_1]
        sin_61 = None
        sin_62 = getitem_249.unsqueeze(1)
        getitem_249 = None
        mul_100 = query_rot_20 * cos_62
        chunk_40 = torch.chunk(query_rot_20, 2, dim=-1)
        query_rot_20 = None
        x1_40 = chunk_40[0]
        x2_40 = chunk_40[1]
        chunk_40 = None
        neg_40 = -x2_40
        x2_40 = None
        cat_80 = torch.cat((neg_40, x1_40), dim=-1)
        neg_40 = x1_40 = None
        mul_101 = cat_80 * sin_62
        cat_80 = None
        q_embed_20 = mul_100 + mul_101
        mul_100 = mul_101 = None
        mul_102 = key_rot_20 * cos_62
        cos_62 = None
        chunk_41 = torch.chunk(key_rot_20, 2, dim=-1)
        key_rot_20 = None
        x1_41 = chunk_41[0]
        x2_41 = chunk_41[1]
        chunk_41 = None
        neg_41 = -x2_41
        x2_41 = None
        cat_81 = torch.cat((neg_41, x1_41), dim=-1)
        neg_41 = x1_41 = None
        mul_103 = cat_81 * sin_62
        cat_81 = sin_62 = None
        k_embed_20 = mul_102 + mul_103
        mul_102 = mul_103 = None
        query_states_62 = torch.cat((q_embed_20, query_pass_20), dim=-1)
        q_embed_20 = query_pass_20 = None
        key_states_62 = torch.cat((k_embed_20, key_pass_20), dim=-1)
        k_embed_20 = key_pass_20 = None
        transpose_103 = key_states_62.transpose(2, 3)
        key_states_62 = None
        matmul_40 = torch.matmul(query_states_62, transpose_103)
        query_states_62 = transpose_103 = None
        attn_weights_60 = matmul_40 / 8.94427190999916
        matmul_40 = None
        attn_weights_61 = attn_weights_60 + combined_attention_mask_1
        attn_weights_60 = None
        softmax_20 = torch.nn.functional.softmax(
            attn_weights_61, dim=-1, dtype=torch.float32
        )
        attn_weights_61 = None
        attn_weights_62 = softmax_20.to(torch.float16)
        softmax_20 = None
        attn_output_80 = torch.matmul(attn_weights_62, value_states_41)
        attn_weights_62 = value_states_41 = None
        transpose_104 = attn_output_80.transpose(1, 2)
        attn_output_80 = None
        attn_output_81 = transpose_104.contiguous()
        transpose_104 = None
        attn_output_82 = attn_output_81.reshape(1, 19, 2560)
        attn_output_81 = None
        attn_output_83 = torch._C._nn.linear(
            attn_output_82,
            l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_82 = l_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_101 = hidden_states_99 + attn_output_83
        hidden_states_99 = attn_output_83 = None
        hidden_states_102 = torch.nn.functional.layer_norm(
            hidden_states_101,
            (2560,),
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_144 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_20 = torch.nn.functional.silu(linear_144, inplace=False)
        linear_144 = None
        linear_145 = torch._C._nn.linear(
            hidden_states_102,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_102 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_104 = silu_20 * linear_145
        silu_20 = linear_145 = None
        hidden_states_103 = torch._C._nn.linear(
            mul_104,
            l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_104 = l_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_104 = hidden_states_101 + hidden_states_103
        hidden_states_101 = hidden_states_103 = None
        hidden_states_105 = torch.nn.functional.layer_norm(
            hidden_states_104,
            (2560,),
            l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_bias_ = (None)
        query_states_63 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_63 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_42 = torch._C._nn.linear(
            hidden_states_105,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_105 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_65 = query_states_63.view(1, 19, 32, 80)
        query_states_63 = None
        query_states_64 = view_65.transpose(1, 2)
        view_65 = None
        view_66 = key_states_63.view(1, 19, 32, 80)
        key_states_63 = None
        key_states_64 = view_66.transpose(1, 2)
        view_66 = None
        view_67 = value_states_42.view(1, 19, 32, 80)
        value_states_42 = None
        value_states_43 = view_67.transpose(1, 2)
        view_67 = None
        query_rot_21 = query_states_64[(Ellipsis, slice(None, 20, None))]
        query_pass_21 = query_states_64[(Ellipsis, slice(20, None, None))]
        query_states_64 = None
        key_rot_21 = key_states_64[(Ellipsis, slice(None, 20, None))]
        key_pass_21 = key_states_64[(Ellipsis, slice(20, None, None))]
        key_states_64 = None
        getitem_258 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_63 = getitem_258.to(dtype=torch.float16)
        getitem_258 = None
        getitem_259 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_63 = getitem_259.to(dtype=torch.float16)
        getitem_259 = None
        squeeze_84 = cos_63.squeeze(1)
        cos_63 = None
        cos_64 = squeeze_84.squeeze(0)
        squeeze_84 = None
        squeeze_86 = sin_63.squeeze(1)
        sin_63 = None
        sin_64 = squeeze_86.squeeze(0)
        squeeze_86 = None
        getitem_260 = cos_64[position_ids_1]
        cos_64 = None
        cos_65 = getitem_260.unsqueeze(1)
        getitem_260 = None
        getitem_261 = sin_64[position_ids_1]
        sin_64 = None
        sin_65 = getitem_261.unsqueeze(1)
        getitem_261 = None
        mul_105 = query_rot_21 * cos_65
        chunk_42 = torch.chunk(query_rot_21, 2, dim=-1)
        query_rot_21 = None
        x1_42 = chunk_42[0]
        x2_42 = chunk_42[1]
        chunk_42 = None
        neg_42 = -x2_42
        x2_42 = None
        cat_84 = torch.cat((neg_42, x1_42), dim=-1)
        neg_42 = x1_42 = None
        mul_106 = cat_84 * sin_65
        cat_84 = None
        q_embed_21 = mul_105 + mul_106
        mul_105 = mul_106 = None
        mul_107 = key_rot_21 * cos_65
        cos_65 = None
        chunk_43 = torch.chunk(key_rot_21, 2, dim=-1)
        key_rot_21 = None
        x1_43 = chunk_43[0]
        x2_43 = chunk_43[1]
        chunk_43 = None
        neg_43 = -x2_43
        x2_43 = None
        cat_85 = torch.cat((neg_43, x1_43), dim=-1)
        neg_43 = x1_43 = None
        mul_108 = cat_85 * sin_65
        cat_85 = sin_65 = None
        k_embed_21 = mul_107 + mul_108
        mul_107 = mul_108 = None
        query_states_65 = torch.cat((q_embed_21, query_pass_21), dim=-1)
        q_embed_21 = query_pass_21 = None
        key_states_65 = torch.cat((k_embed_21, key_pass_21), dim=-1)
        k_embed_21 = key_pass_21 = None
        transpose_108 = key_states_65.transpose(2, 3)
        key_states_65 = None
        matmul_42 = torch.matmul(query_states_65, transpose_108)
        query_states_65 = transpose_108 = None
        attn_weights_63 = matmul_42 / 8.94427190999916
        matmul_42 = None
        attn_weights_64 = attn_weights_63 + combined_attention_mask_1
        attn_weights_63 = None
        softmax_21 = torch.nn.functional.softmax(
            attn_weights_64, dim=-1, dtype=torch.float32
        )
        attn_weights_64 = None
        attn_weights_65 = softmax_21.to(torch.float16)
        softmax_21 = None
        attn_output_84 = torch.matmul(attn_weights_65, value_states_43)
        attn_weights_65 = value_states_43 = None
        transpose_109 = attn_output_84.transpose(1, 2)
        attn_output_84 = None
        attn_output_85 = transpose_109.contiguous()
        transpose_109 = None
        attn_output_86 = attn_output_85.reshape(1, 19, 2560)
        attn_output_85 = None
        attn_output_87 = torch._C._nn.linear(
            attn_output_86,
            l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_86 = l_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_106 = hidden_states_104 + attn_output_87
        hidden_states_104 = attn_output_87 = None
        hidden_states_107 = torch.nn.functional.layer_norm(
            hidden_states_106,
            (2560,),
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_151 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_21 = torch.nn.functional.silu(linear_151, inplace=False)
        linear_151 = None
        linear_152 = torch._C._nn.linear(
            hidden_states_107,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_107 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_109 = silu_21 * linear_152
        silu_21 = linear_152 = None
        hidden_states_108 = torch._C._nn.linear(
            mul_109,
            l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_109 = l_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_109 = hidden_states_106 + hidden_states_108
        hidden_states_106 = hidden_states_108 = None
        hidden_states_110 = torch.nn.functional.layer_norm(
            hidden_states_109,
            (2560,),
            l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_bias_ = (None)
        query_states_66 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_66 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_44 = torch._C._nn.linear(
            hidden_states_110,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_110 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_68 = query_states_66.view(1, 19, 32, 80)
        query_states_66 = None
        query_states_67 = view_68.transpose(1, 2)
        view_68 = None
        view_69 = key_states_66.view(1, 19, 32, 80)
        key_states_66 = None
        key_states_67 = view_69.transpose(1, 2)
        view_69 = None
        view_70 = value_states_44.view(1, 19, 32, 80)
        value_states_44 = None
        value_states_45 = view_70.transpose(1, 2)
        view_70 = None
        query_rot_22 = query_states_67[(Ellipsis, slice(None, 20, None))]
        query_pass_22 = query_states_67[(Ellipsis, slice(20, None, None))]
        query_states_67 = None
        key_rot_22 = key_states_67[(Ellipsis, slice(None, 20, None))]
        key_pass_22 = key_states_67[(Ellipsis, slice(20, None, None))]
        key_states_67 = None
        getitem_270 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_66 = getitem_270.to(dtype=torch.float16)
        getitem_270 = None
        getitem_271 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_66 = getitem_271.to(dtype=torch.float16)
        getitem_271 = None
        squeeze_88 = cos_66.squeeze(1)
        cos_66 = None
        cos_67 = squeeze_88.squeeze(0)
        squeeze_88 = None
        squeeze_90 = sin_66.squeeze(1)
        sin_66 = None
        sin_67 = squeeze_90.squeeze(0)
        squeeze_90 = None
        getitem_272 = cos_67[position_ids_1]
        cos_67 = None
        cos_68 = getitem_272.unsqueeze(1)
        getitem_272 = None
        getitem_273 = sin_67[position_ids_1]
        sin_67 = None
        sin_68 = getitem_273.unsqueeze(1)
        getitem_273 = None
        mul_110 = query_rot_22 * cos_68
        chunk_44 = torch.chunk(query_rot_22, 2, dim=-1)
        query_rot_22 = None
        x1_44 = chunk_44[0]
        x2_44 = chunk_44[1]
        chunk_44 = None
        neg_44 = -x2_44
        x2_44 = None
        cat_88 = torch.cat((neg_44, x1_44), dim=-1)
        neg_44 = x1_44 = None
        mul_111 = cat_88 * sin_68
        cat_88 = None
        q_embed_22 = mul_110 + mul_111
        mul_110 = mul_111 = None
        mul_112 = key_rot_22 * cos_68
        cos_68 = None
        chunk_45 = torch.chunk(key_rot_22, 2, dim=-1)
        key_rot_22 = None
        x1_45 = chunk_45[0]
        x2_45 = chunk_45[1]
        chunk_45 = None
        neg_45 = -x2_45
        x2_45 = None
        cat_89 = torch.cat((neg_45, x1_45), dim=-1)
        neg_45 = x1_45 = None
        mul_113 = cat_89 * sin_68
        cat_89 = sin_68 = None
        k_embed_22 = mul_112 + mul_113
        mul_112 = mul_113 = None
        query_states_68 = torch.cat((q_embed_22, query_pass_22), dim=-1)
        q_embed_22 = query_pass_22 = None
        key_states_68 = torch.cat((k_embed_22, key_pass_22), dim=-1)
        k_embed_22 = key_pass_22 = None
        transpose_113 = key_states_68.transpose(2, 3)
        key_states_68 = None
        matmul_44 = torch.matmul(query_states_68, transpose_113)
        query_states_68 = transpose_113 = None
        attn_weights_66 = matmul_44 / 8.94427190999916
        matmul_44 = None
        attn_weights_67 = attn_weights_66 + combined_attention_mask_1
        attn_weights_66 = None
        softmax_22 = torch.nn.functional.softmax(
            attn_weights_67, dim=-1, dtype=torch.float32
        )
        attn_weights_67 = None
        attn_weights_68 = softmax_22.to(torch.float16)
        softmax_22 = None
        attn_output_88 = torch.matmul(attn_weights_68, value_states_45)
        attn_weights_68 = value_states_45 = None
        transpose_114 = attn_output_88.transpose(1, 2)
        attn_output_88 = None
        attn_output_89 = transpose_114.contiguous()
        transpose_114 = None
        attn_output_90 = attn_output_89.reshape(1, 19, 2560)
        attn_output_89 = None
        attn_output_91 = torch._C._nn.linear(
            attn_output_90,
            l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_90 = l_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_111 = hidden_states_109 + attn_output_91
        hidden_states_109 = attn_output_91 = None
        hidden_states_112 = torch.nn.functional.layer_norm(
            hidden_states_111,
            (2560,),
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_158 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_22 = torch.nn.functional.silu(linear_158, inplace=False)
        linear_158 = None
        linear_159 = torch._C._nn.linear(
            hidden_states_112,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_112 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_114 = silu_22 * linear_159
        silu_22 = linear_159 = None
        hidden_states_113 = torch._C._nn.linear(
            mul_114,
            l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_114 = l_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_114 = hidden_states_111 + hidden_states_113
        hidden_states_111 = hidden_states_113 = None
        hidden_states_115 = torch.nn.functional.layer_norm(
            hidden_states_114,
            (2560,),
            l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_bias_ = (None)
        query_states_69 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_69 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_46 = torch._C._nn.linear(
            hidden_states_115,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_115 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_71 = query_states_69.view(1, 19, 32, 80)
        query_states_69 = None
        query_states_70 = view_71.transpose(1, 2)
        view_71 = None
        view_72 = key_states_69.view(1, 19, 32, 80)
        key_states_69 = None
        key_states_70 = view_72.transpose(1, 2)
        view_72 = None
        view_73 = value_states_46.view(1, 19, 32, 80)
        value_states_46 = None
        value_states_47 = view_73.transpose(1, 2)
        view_73 = None
        query_rot_23 = query_states_70[(Ellipsis, slice(None, 20, None))]
        query_pass_23 = query_states_70[(Ellipsis, slice(20, None, None))]
        query_states_70 = None
        key_rot_23 = key_states_70[(Ellipsis, slice(None, 20, None))]
        key_pass_23 = key_states_70[(Ellipsis, slice(20, None, None))]
        key_states_70 = None
        getitem_282 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_69 = getitem_282.to(dtype=torch.float16)
        getitem_282 = None
        getitem_283 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_69 = getitem_283.to(dtype=torch.float16)
        getitem_283 = None
        squeeze_92 = cos_69.squeeze(1)
        cos_69 = None
        cos_70 = squeeze_92.squeeze(0)
        squeeze_92 = None
        squeeze_94 = sin_69.squeeze(1)
        sin_69 = None
        sin_70 = squeeze_94.squeeze(0)
        squeeze_94 = None
        getitem_284 = cos_70[position_ids_1]
        cos_70 = None
        cos_71 = getitem_284.unsqueeze(1)
        getitem_284 = None
        getitem_285 = sin_70[position_ids_1]
        sin_70 = None
        sin_71 = getitem_285.unsqueeze(1)
        getitem_285 = None
        mul_115 = query_rot_23 * cos_71
        chunk_46 = torch.chunk(query_rot_23, 2, dim=-1)
        query_rot_23 = None
        x1_46 = chunk_46[0]
        x2_46 = chunk_46[1]
        chunk_46 = None
        neg_46 = -x2_46
        x2_46 = None
        cat_92 = torch.cat((neg_46, x1_46), dim=-1)
        neg_46 = x1_46 = None
        mul_116 = cat_92 * sin_71
        cat_92 = None
        q_embed_23 = mul_115 + mul_116
        mul_115 = mul_116 = None
        mul_117 = key_rot_23 * cos_71
        cos_71 = None
        chunk_47 = torch.chunk(key_rot_23, 2, dim=-1)
        key_rot_23 = None
        x1_47 = chunk_47[0]
        x2_47 = chunk_47[1]
        chunk_47 = None
        neg_47 = -x2_47
        x2_47 = None
        cat_93 = torch.cat((neg_47, x1_47), dim=-1)
        neg_47 = x1_47 = None
        mul_118 = cat_93 * sin_71
        cat_93 = sin_71 = None
        k_embed_23 = mul_117 + mul_118
        mul_117 = mul_118 = None
        query_states_71 = torch.cat((q_embed_23, query_pass_23), dim=-1)
        q_embed_23 = query_pass_23 = None
        key_states_71 = torch.cat((k_embed_23, key_pass_23), dim=-1)
        k_embed_23 = key_pass_23 = None
        transpose_118 = key_states_71.transpose(2, 3)
        key_states_71 = None
        matmul_46 = torch.matmul(query_states_71, transpose_118)
        query_states_71 = transpose_118 = None
        attn_weights_69 = matmul_46 / 8.94427190999916
        matmul_46 = None
        attn_weights_70 = attn_weights_69 + combined_attention_mask_1
        attn_weights_69 = None
        softmax_23 = torch.nn.functional.softmax(
            attn_weights_70, dim=-1, dtype=torch.float32
        )
        attn_weights_70 = None
        attn_weights_71 = softmax_23.to(torch.float16)
        softmax_23 = None
        attn_output_92 = torch.matmul(attn_weights_71, value_states_47)
        attn_weights_71 = value_states_47 = None
        transpose_119 = attn_output_92.transpose(1, 2)
        attn_output_92 = None
        attn_output_93 = transpose_119.contiguous()
        transpose_119 = None
        attn_output_94 = attn_output_93.reshape(1, 19, 2560)
        attn_output_93 = None
        attn_output_95 = torch._C._nn.linear(
            attn_output_94,
            l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_94 = l_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_116 = hidden_states_114 + attn_output_95
        hidden_states_114 = attn_output_95 = None
        hidden_states_117 = torch.nn.functional.layer_norm(
            hidden_states_116,
            (2560,),
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_165 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_23 = torch.nn.functional.silu(linear_165, inplace=False)
        linear_165 = None
        linear_166 = torch._C._nn.linear(
            hidden_states_117,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_117 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_119 = silu_23 * linear_166
        silu_23 = linear_166 = None
        hidden_states_118 = torch._C._nn.linear(
            mul_119,
            l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_119 = l_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_119 = hidden_states_116 + hidden_states_118
        hidden_states_116 = hidden_states_118 = None
        hidden_states_120 = torch.nn.functional.layer_norm(
            hidden_states_119,
            (2560,),
            l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_bias_ = (None)
        query_states_72 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_72 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_48 = torch._C._nn.linear(
            hidden_states_120,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_120 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_74 = query_states_72.view(1, 19, 32, 80)
        query_states_72 = None
        query_states_73 = view_74.transpose(1, 2)
        view_74 = None
        view_75 = key_states_72.view(1, 19, 32, 80)
        key_states_72 = None
        key_states_73 = view_75.transpose(1, 2)
        view_75 = None
        view_76 = value_states_48.view(1, 19, 32, 80)
        value_states_48 = None
        value_states_49 = view_76.transpose(1, 2)
        view_76 = None
        query_rot_24 = query_states_73[(Ellipsis, slice(None, 20, None))]
        query_pass_24 = query_states_73[(Ellipsis, slice(20, None, None))]
        query_states_73 = None
        key_rot_24 = key_states_73[(Ellipsis, slice(None, 20, None))]
        key_pass_24 = key_states_73[(Ellipsis, slice(20, None, None))]
        key_states_73 = None
        getitem_294 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_72 = getitem_294.to(dtype=torch.float16)
        getitem_294 = None
        getitem_295 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_72 = getitem_295.to(dtype=torch.float16)
        getitem_295 = None
        squeeze_96 = cos_72.squeeze(1)
        cos_72 = None
        cos_73 = squeeze_96.squeeze(0)
        squeeze_96 = None
        squeeze_98 = sin_72.squeeze(1)
        sin_72 = None
        sin_73 = squeeze_98.squeeze(0)
        squeeze_98 = None
        getitem_296 = cos_73[position_ids_1]
        cos_73 = None
        cos_74 = getitem_296.unsqueeze(1)
        getitem_296 = None
        getitem_297 = sin_73[position_ids_1]
        sin_73 = None
        sin_74 = getitem_297.unsqueeze(1)
        getitem_297 = None
        mul_120 = query_rot_24 * cos_74
        chunk_48 = torch.chunk(query_rot_24, 2, dim=-1)
        query_rot_24 = None
        x1_48 = chunk_48[0]
        x2_48 = chunk_48[1]
        chunk_48 = None
        neg_48 = -x2_48
        x2_48 = None
        cat_96 = torch.cat((neg_48, x1_48), dim=-1)
        neg_48 = x1_48 = None
        mul_121 = cat_96 * sin_74
        cat_96 = None
        q_embed_24 = mul_120 + mul_121
        mul_120 = mul_121 = None
        mul_122 = key_rot_24 * cos_74
        cos_74 = None
        chunk_49 = torch.chunk(key_rot_24, 2, dim=-1)
        key_rot_24 = None
        x1_49 = chunk_49[0]
        x2_49 = chunk_49[1]
        chunk_49 = None
        neg_49 = -x2_49
        x2_49 = None
        cat_97 = torch.cat((neg_49, x1_49), dim=-1)
        neg_49 = x1_49 = None
        mul_123 = cat_97 * sin_74
        cat_97 = sin_74 = None
        k_embed_24 = mul_122 + mul_123
        mul_122 = mul_123 = None
        query_states_74 = torch.cat((q_embed_24, query_pass_24), dim=-1)
        q_embed_24 = query_pass_24 = None
        key_states_74 = torch.cat((k_embed_24, key_pass_24), dim=-1)
        k_embed_24 = key_pass_24 = None
        transpose_123 = key_states_74.transpose(2, 3)
        key_states_74 = None
        matmul_48 = torch.matmul(query_states_74, transpose_123)
        query_states_74 = transpose_123 = None
        attn_weights_72 = matmul_48 / 8.94427190999916
        matmul_48 = None
        attn_weights_73 = attn_weights_72 + combined_attention_mask_1
        attn_weights_72 = None
        softmax_24 = torch.nn.functional.softmax(
            attn_weights_73, dim=-1, dtype=torch.float32
        )
        attn_weights_73 = None
        attn_weights_74 = softmax_24.to(torch.float16)
        softmax_24 = None
        attn_output_96 = torch.matmul(attn_weights_74, value_states_49)
        attn_weights_74 = value_states_49 = None
        transpose_124 = attn_output_96.transpose(1, 2)
        attn_output_96 = None
        attn_output_97 = transpose_124.contiguous()
        transpose_124 = None
        attn_output_98 = attn_output_97.reshape(1, 19, 2560)
        attn_output_97 = None
        attn_output_99 = torch._C._nn.linear(
            attn_output_98,
            l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_98 = l_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_121 = hidden_states_119 + attn_output_99
        hidden_states_119 = attn_output_99 = None
        hidden_states_122 = torch.nn.functional.layer_norm(
            hidden_states_121,
            (2560,),
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_172 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_24 = torch.nn.functional.silu(linear_172, inplace=False)
        linear_172 = None
        linear_173 = torch._C._nn.linear(
            hidden_states_122,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_122 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_124 = silu_24 * linear_173
        silu_24 = linear_173 = None
        hidden_states_123 = torch._C._nn.linear(
            mul_124,
            l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_124 = l_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_124 = hidden_states_121 + hidden_states_123
        hidden_states_121 = hidden_states_123 = None
        hidden_states_125 = torch.nn.functional.layer_norm(
            hidden_states_124,
            (2560,),
            l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_bias_ = (None)
        query_states_75 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_75 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_50 = torch._C._nn.linear(
            hidden_states_125,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_125 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_77 = query_states_75.view(1, 19, 32, 80)
        query_states_75 = None
        query_states_76 = view_77.transpose(1, 2)
        view_77 = None
        view_78 = key_states_75.view(1, 19, 32, 80)
        key_states_75 = None
        key_states_76 = view_78.transpose(1, 2)
        view_78 = None
        view_79 = value_states_50.view(1, 19, 32, 80)
        value_states_50 = None
        value_states_51 = view_79.transpose(1, 2)
        view_79 = None
        query_rot_25 = query_states_76[(Ellipsis, slice(None, 20, None))]
        query_pass_25 = query_states_76[(Ellipsis, slice(20, None, None))]
        query_states_76 = None
        key_rot_25 = key_states_76[(Ellipsis, slice(None, 20, None))]
        key_pass_25 = key_states_76[(Ellipsis, slice(20, None, None))]
        key_states_76 = None
        getitem_306 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_75 = getitem_306.to(dtype=torch.float16)
        getitem_306 = None
        getitem_307 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_75 = getitem_307.to(dtype=torch.float16)
        getitem_307 = None
        squeeze_100 = cos_75.squeeze(1)
        cos_75 = None
        cos_76 = squeeze_100.squeeze(0)
        squeeze_100 = None
        squeeze_102 = sin_75.squeeze(1)
        sin_75 = None
        sin_76 = squeeze_102.squeeze(0)
        squeeze_102 = None
        getitem_308 = cos_76[position_ids_1]
        cos_76 = None
        cos_77 = getitem_308.unsqueeze(1)
        getitem_308 = None
        getitem_309 = sin_76[position_ids_1]
        sin_76 = None
        sin_77 = getitem_309.unsqueeze(1)
        getitem_309 = None
        mul_125 = query_rot_25 * cos_77
        chunk_50 = torch.chunk(query_rot_25, 2, dim=-1)
        query_rot_25 = None
        x1_50 = chunk_50[0]
        x2_50 = chunk_50[1]
        chunk_50 = None
        neg_50 = -x2_50
        x2_50 = None
        cat_100 = torch.cat((neg_50, x1_50), dim=-1)
        neg_50 = x1_50 = None
        mul_126 = cat_100 * sin_77
        cat_100 = None
        q_embed_25 = mul_125 + mul_126
        mul_125 = mul_126 = None
        mul_127 = key_rot_25 * cos_77
        cos_77 = None
        chunk_51 = torch.chunk(key_rot_25, 2, dim=-1)
        key_rot_25 = None
        x1_51 = chunk_51[0]
        x2_51 = chunk_51[1]
        chunk_51 = None
        neg_51 = -x2_51
        x2_51 = None
        cat_101 = torch.cat((neg_51, x1_51), dim=-1)
        neg_51 = x1_51 = None
        mul_128 = cat_101 * sin_77
        cat_101 = sin_77 = None
        k_embed_25 = mul_127 + mul_128
        mul_127 = mul_128 = None
        query_states_77 = torch.cat((q_embed_25, query_pass_25), dim=-1)
        q_embed_25 = query_pass_25 = None
        key_states_77 = torch.cat((k_embed_25, key_pass_25), dim=-1)
        k_embed_25 = key_pass_25 = None
        transpose_128 = key_states_77.transpose(2, 3)
        key_states_77 = None
        matmul_50 = torch.matmul(query_states_77, transpose_128)
        query_states_77 = transpose_128 = None
        attn_weights_75 = matmul_50 / 8.94427190999916
        matmul_50 = None
        attn_weights_76 = attn_weights_75 + combined_attention_mask_1
        attn_weights_75 = None
        softmax_25 = torch.nn.functional.softmax(
            attn_weights_76, dim=-1, dtype=torch.float32
        )
        attn_weights_76 = None
        attn_weights_77 = softmax_25.to(torch.float16)
        softmax_25 = None
        attn_output_100 = torch.matmul(attn_weights_77, value_states_51)
        attn_weights_77 = value_states_51 = None
        transpose_129 = attn_output_100.transpose(1, 2)
        attn_output_100 = None
        attn_output_101 = transpose_129.contiguous()
        transpose_129 = None
        attn_output_102 = attn_output_101.reshape(1, 19, 2560)
        attn_output_101 = None
        attn_output_103 = torch._C._nn.linear(
            attn_output_102,
            l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_102 = l_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_126 = hidden_states_124 + attn_output_103
        hidden_states_124 = attn_output_103 = None
        hidden_states_127 = torch.nn.functional.layer_norm(
            hidden_states_126,
            (2560,),
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_179 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_25 = torch.nn.functional.silu(linear_179, inplace=False)
        linear_179 = None
        linear_180 = torch._C._nn.linear(
            hidden_states_127,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_127 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_129 = silu_25 * linear_180
        silu_25 = linear_180 = None
        hidden_states_128 = torch._C._nn.linear(
            mul_129,
            l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_129 = l_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_129 = hidden_states_126 + hidden_states_128
        hidden_states_126 = hidden_states_128 = None
        hidden_states_130 = torch.nn.functional.layer_norm(
            hidden_states_129,
            (2560,),
            l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_26_modules_input_layernorm_parameters_bias_ = (None)
        query_states_78 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_78 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_52 = torch._C._nn.linear(
            hidden_states_130,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_130 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_80 = query_states_78.view(1, 19, 32, 80)
        query_states_78 = None
        query_states_79 = view_80.transpose(1, 2)
        view_80 = None
        view_81 = key_states_78.view(1, 19, 32, 80)
        key_states_78 = None
        key_states_79 = view_81.transpose(1, 2)
        view_81 = None
        view_82 = value_states_52.view(1, 19, 32, 80)
        value_states_52 = None
        value_states_53 = view_82.transpose(1, 2)
        view_82 = None
        query_rot_26 = query_states_79[(Ellipsis, slice(None, 20, None))]
        query_pass_26 = query_states_79[(Ellipsis, slice(20, None, None))]
        query_states_79 = None
        key_rot_26 = key_states_79[(Ellipsis, slice(None, 20, None))]
        key_pass_26 = key_states_79[(Ellipsis, slice(20, None, None))]
        key_states_79 = None
        getitem_318 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_78 = getitem_318.to(dtype=torch.float16)
        getitem_318 = None
        getitem_319 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_78 = getitem_319.to(dtype=torch.float16)
        getitem_319 = None
        squeeze_104 = cos_78.squeeze(1)
        cos_78 = None
        cos_79 = squeeze_104.squeeze(0)
        squeeze_104 = None
        squeeze_106 = sin_78.squeeze(1)
        sin_78 = None
        sin_79 = squeeze_106.squeeze(0)
        squeeze_106 = None
        getitem_320 = cos_79[position_ids_1]
        cos_79 = None
        cos_80 = getitem_320.unsqueeze(1)
        getitem_320 = None
        getitem_321 = sin_79[position_ids_1]
        sin_79 = None
        sin_80 = getitem_321.unsqueeze(1)
        getitem_321 = None
        mul_130 = query_rot_26 * cos_80
        chunk_52 = torch.chunk(query_rot_26, 2, dim=-1)
        query_rot_26 = None
        x1_52 = chunk_52[0]
        x2_52 = chunk_52[1]
        chunk_52 = None
        neg_52 = -x2_52
        x2_52 = None
        cat_104 = torch.cat((neg_52, x1_52), dim=-1)
        neg_52 = x1_52 = None
        mul_131 = cat_104 * sin_80
        cat_104 = None
        q_embed_26 = mul_130 + mul_131
        mul_130 = mul_131 = None
        mul_132 = key_rot_26 * cos_80
        cos_80 = None
        chunk_53 = torch.chunk(key_rot_26, 2, dim=-1)
        key_rot_26 = None
        x1_53 = chunk_53[0]
        x2_53 = chunk_53[1]
        chunk_53 = None
        neg_53 = -x2_53
        x2_53 = None
        cat_105 = torch.cat((neg_53, x1_53), dim=-1)
        neg_53 = x1_53 = None
        mul_133 = cat_105 * sin_80
        cat_105 = sin_80 = None
        k_embed_26 = mul_132 + mul_133
        mul_132 = mul_133 = None
        query_states_80 = torch.cat((q_embed_26, query_pass_26), dim=-1)
        q_embed_26 = query_pass_26 = None
        key_states_80 = torch.cat((k_embed_26, key_pass_26), dim=-1)
        k_embed_26 = key_pass_26 = None
        transpose_133 = key_states_80.transpose(2, 3)
        key_states_80 = None
        matmul_52 = torch.matmul(query_states_80, transpose_133)
        query_states_80 = transpose_133 = None
        attn_weights_78 = matmul_52 / 8.94427190999916
        matmul_52 = None
        attn_weights_79 = attn_weights_78 + combined_attention_mask_1
        attn_weights_78 = None
        softmax_26 = torch.nn.functional.softmax(
            attn_weights_79, dim=-1, dtype=torch.float32
        )
        attn_weights_79 = None
        attn_weights_80 = softmax_26.to(torch.float16)
        softmax_26 = None
        attn_output_104 = torch.matmul(attn_weights_80, value_states_53)
        attn_weights_80 = value_states_53 = None
        transpose_134 = attn_output_104.transpose(1, 2)
        attn_output_104 = None
        attn_output_105 = transpose_134.contiguous()
        transpose_134 = None
        attn_output_106 = attn_output_105.reshape(1, 19, 2560)
        attn_output_105 = None
        attn_output_107 = torch._C._nn.linear(
            attn_output_106,
            l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_106 = l_self_modules_model_modules_layers_modules_26_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_131 = hidden_states_129 + attn_output_107
        hidden_states_129 = attn_output_107 = None
        hidden_states_132 = torch.nn.functional.layer_norm(
            hidden_states_131,
            (2560,),
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_26_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_186 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_26 = torch.nn.functional.silu(linear_186, inplace=False)
        linear_186 = None
        linear_187 = torch._C._nn.linear(
            hidden_states_132,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_132 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_134 = silu_26 * linear_187
        silu_26 = linear_187 = None
        hidden_states_133 = torch._C._nn.linear(
            mul_134,
            l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_134 = l_self_modules_model_modules_layers_modules_26_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_134 = hidden_states_131 + hidden_states_133
        hidden_states_131 = hidden_states_133 = None
        hidden_states_135 = torch.nn.functional.layer_norm(
            hidden_states_134,
            (2560,),
            l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_27_modules_input_layernorm_parameters_bias_ = (None)
        query_states_81 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_81 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_54 = torch._C._nn.linear(
            hidden_states_135,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_135 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_83 = query_states_81.view(1, 19, 32, 80)
        query_states_81 = None
        query_states_82 = view_83.transpose(1, 2)
        view_83 = None
        view_84 = key_states_81.view(1, 19, 32, 80)
        key_states_81 = None
        key_states_82 = view_84.transpose(1, 2)
        view_84 = None
        view_85 = value_states_54.view(1, 19, 32, 80)
        value_states_54 = None
        value_states_55 = view_85.transpose(1, 2)
        view_85 = None
        query_rot_27 = query_states_82[(Ellipsis, slice(None, 20, None))]
        query_pass_27 = query_states_82[(Ellipsis, slice(20, None, None))]
        query_states_82 = None
        key_rot_27 = key_states_82[(Ellipsis, slice(None, 20, None))]
        key_pass_27 = key_states_82[(Ellipsis, slice(20, None, None))]
        key_states_82 = None
        getitem_330 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_81 = getitem_330.to(dtype=torch.float16)
        getitem_330 = None
        getitem_331 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_81 = getitem_331.to(dtype=torch.float16)
        getitem_331 = None
        squeeze_108 = cos_81.squeeze(1)
        cos_81 = None
        cos_82 = squeeze_108.squeeze(0)
        squeeze_108 = None
        squeeze_110 = sin_81.squeeze(1)
        sin_81 = None
        sin_82 = squeeze_110.squeeze(0)
        squeeze_110 = None
        getitem_332 = cos_82[position_ids_1]
        cos_82 = None
        cos_83 = getitem_332.unsqueeze(1)
        getitem_332 = None
        getitem_333 = sin_82[position_ids_1]
        sin_82 = None
        sin_83 = getitem_333.unsqueeze(1)
        getitem_333 = None
        mul_135 = query_rot_27 * cos_83
        chunk_54 = torch.chunk(query_rot_27, 2, dim=-1)
        query_rot_27 = None
        x1_54 = chunk_54[0]
        x2_54 = chunk_54[1]
        chunk_54 = None
        neg_54 = -x2_54
        x2_54 = None
        cat_108 = torch.cat((neg_54, x1_54), dim=-1)
        neg_54 = x1_54 = None
        mul_136 = cat_108 * sin_83
        cat_108 = None
        q_embed_27 = mul_135 + mul_136
        mul_135 = mul_136 = None
        mul_137 = key_rot_27 * cos_83
        cos_83 = None
        chunk_55 = torch.chunk(key_rot_27, 2, dim=-1)
        key_rot_27 = None
        x1_55 = chunk_55[0]
        x2_55 = chunk_55[1]
        chunk_55 = None
        neg_55 = -x2_55
        x2_55 = None
        cat_109 = torch.cat((neg_55, x1_55), dim=-1)
        neg_55 = x1_55 = None
        mul_138 = cat_109 * sin_83
        cat_109 = sin_83 = None
        k_embed_27 = mul_137 + mul_138
        mul_137 = mul_138 = None
        query_states_83 = torch.cat((q_embed_27, query_pass_27), dim=-1)
        q_embed_27 = query_pass_27 = None
        key_states_83 = torch.cat((k_embed_27, key_pass_27), dim=-1)
        k_embed_27 = key_pass_27 = None
        transpose_138 = key_states_83.transpose(2, 3)
        key_states_83 = None
        matmul_54 = torch.matmul(query_states_83, transpose_138)
        query_states_83 = transpose_138 = None
        attn_weights_81 = matmul_54 / 8.94427190999916
        matmul_54 = None
        attn_weights_82 = attn_weights_81 + combined_attention_mask_1
        attn_weights_81 = None
        softmax_27 = torch.nn.functional.softmax(
            attn_weights_82, dim=-1, dtype=torch.float32
        )
        attn_weights_82 = None
        attn_weights_83 = softmax_27.to(torch.float16)
        softmax_27 = None
        attn_output_108 = torch.matmul(attn_weights_83, value_states_55)
        attn_weights_83 = value_states_55 = None
        transpose_139 = attn_output_108.transpose(1, 2)
        attn_output_108 = None
        attn_output_109 = transpose_139.contiguous()
        transpose_139 = None
        attn_output_110 = attn_output_109.reshape(1, 19, 2560)
        attn_output_109 = None
        attn_output_111 = torch._C._nn.linear(
            attn_output_110,
            l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_110 = l_self_modules_model_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_136 = hidden_states_134 + attn_output_111
        hidden_states_134 = attn_output_111 = None
        hidden_states_137 = torch.nn.functional.layer_norm(
            hidden_states_136,
            (2560,),
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_27_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_193 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_27 = torch.nn.functional.silu(linear_193, inplace=False)
        linear_193 = None
        linear_194 = torch._C._nn.linear(
            hidden_states_137,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_137 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_139 = silu_27 * linear_194
        silu_27 = linear_194 = None
        hidden_states_138 = torch._C._nn.linear(
            mul_139,
            l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_139 = l_self_modules_model_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_139 = hidden_states_136 + hidden_states_138
        hidden_states_136 = hidden_states_138 = None
        hidden_states_140 = torch.nn.functional.layer_norm(
            hidden_states_139,
            (2560,),
            l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_28_modules_input_layernorm_parameters_bias_ = (None)
        query_states_84 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_84 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_56 = torch._C._nn.linear(
            hidden_states_140,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_140 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_86 = query_states_84.view(1, 19, 32, 80)
        query_states_84 = None
        query_states_85 = view_86.transpose(1, 2)
        view_86 = None
        view_87 = key_states_84.view(1, 19, 32, 80)
        key_states_84 = None
        key_states_85 = view_87.transpose(1, 2)
        view_87 = None
        view_88 = value_states_56.view(1, 19, 32, 80)
        value_states_56 = None
        value_states_57 = view_88.transpose(1, 2)
        view_88 = None
        query_rot_28 = query_states_85[(Ellipsis, slice(None, 20, None))]
        query_pass_28 = query_states_85[(Ellipsis, slice(20, None, None))]
        query_states_85 = None
        key_rot_28 = key_states_85[(Ellipsis, slice(None, 20, None))]
        key_pass_28 = key_states_85[(Ellipsis, slice(20, None, None))]
        key_states_85 = None
        getitem_342 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_84 = getitem_342.to(dtype=torch.float16)
        getitem_342 = None
        getitem_343 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_84 = getitem_343.to(dtype=torch.float16)
        getitem_343 = None
        squeeze_112 = cos_84.squeeze(1)
        cos_84 = None
        cos_85 = squeeze_112.squeeze(0)
        squeeze_112 = None
        squeeze_114 = sin_84.squeeze(1)
        sin_84 = None
        sin_85 = squeeze_114.squeeze(0)
        squeeze_114 = None
        getitem_344 = cos_85[position_ids_1]
        cos_85 = None
        cos_86 = getitem_344.unsqueeze(1)
        getitem_344 = None
        getitem_345 = sin_85[position_ids_1]
        sin_85 = None
        sin_86 = getitem_345.unsqueeze(1)
        getitem_345 = None
        mul_140 = query_rot_28 * cos_86
        chunk_56 = torch.chunk(query_rot_28, 2, dim=-1)
        query_rot_28 = None
        x1_56 = chunk_56[0]
        x2_56 = chunk_56[1]
        chunk_56 = None
        neg_56 = -x2_56
        x2_56 = None
        cat_112 = torch.cat((neg_56, x1_56), dim=-1)
        neg_56 = x1_56 = None
        mul_141 = cat_112 * sin_86
        cat_112 = None
        q_embed_28 = mul_140 + mul_141
        mul_140 = mul_141 = None
        mul_142 = key_rot_28 * cos_86
        cos_86 = None
        chunk_57 = torch.chunk(key_rot_28, 2, dim=-1)
        key_rot_28 = None
        x1_57 = chunk_57[0]
        x2_57 = chunk_57[1]
        chunk_57 = None
        neg_57 = -x2_57
        x2_57 = None
        cat_113 = torch.cat((neg_57, x1_57), dim=-1)
        neg_57 = x1_57 = None
        mul_143 = cat_113 * sin_86
        cat_113 = sin_86 = None
        k_embed_28 = mul_142 + mul_143
        mul_142 = mul_143 = None
        query_states_86 = torch.cat((q_embed_28, query_pass_28), dim=-1)
        q_embed_28 = query_pass_28 = None
        key_states_86 = torch.cat((k_embed_28, key_pass_28), dim=-1)
        k_embed_28 = key_pass_28 = None
        transpose_143 = key_states_86.transpose(2, 3)
        key_states_86 = None
        matmul_56 = torch.matmul(query_states_86, transpose_143)
        query_states_86 = transpose_143 = None
        attn_weights_84 = matmul_56 / 8.94427190999916
        matmul_56 = None
        attn_weights_85 = attn_weights_84 + combined_attention_mask_1
        attn_weights_84 = None
        softmax_28 = torch.nn.functional.softmax(
            attn_weights_85, dim=-1, dtype=torch.float32
        )
        attn_weights_85 = None
        attn_weights_86 = softmax_28.to(torch.float16)
        softmax_28 = None
        attn_output_112 = torch.matmul(attn_weights_86, value_states_57)
        attn_weights_86 = value_states_57 = None
        transpose_144 = attn_output_112.transpose(1, 2)
        attn_output_112 = None
        attn_output_113 = transpose_144.contiguous()
        transpose_144 = None
        attn_output_114 = attn_output_113.reshape(1, 19, 2560)
        attn_output_113 = None
        attn_output_115 = torch._C._nn.linear(
            attn_output_114,
            l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_114 = l_self_modules_model_modules_layers_modules_28_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_141 = hidden_states_139 + attn_output_115
        hidden_states_139 = attn_output_115 = None
        hidden_states_142 = torch.nn.functional.layer_norm(
            hidden_states_141,
            (2560,),
            l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_28_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_200 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_28 = torch.nn.functional.silu(linear_200, inplace=False)
        linear_200 = None
        linear_201 = torch._C._nn.linear(
            hidden_states_142,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_142 = l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_144 = silu_28 * linear_201
        silu_28 = linear_201 = None
        hidden_states_143 = torch._C._nn.linear(
            mul_144,
            l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_144 = l_self_modules_model_modules_layers_modules_28_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_144 = hidden_states_141 + hidden_states_143
        hidden_states_141 = hidden_states_143 = None
        hidden_states_145 = torch.nn.functional.layer_norm(
            hidden_states_144,
            (2560,),
            l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_29_modules_input_layernorm_parameters_bias_ = (None)
        query_states_87 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_87 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_58 = torch._C._nn.linear(
            hidden_states_145,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_145 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_89 = query_states_87.view(1, 19, 32, 80)
        query_states_87 = None
        query_states_88 = view_89.transpose(1, 2)
        view_89 = None
        view_90 = key_states_87.view(1, 19, 32, 80)
        key_states_87 = None
        key_states_88 = view_90.transpose(1, 2)
        view_90 = None
        view_91 = value_states_58.view(1, 19, 32, 80)
        value_states_58 = None
        value_states_59 = view_91.transpose(1, 2)
        view_91 = None
        query_rot_29 = query_states_88[(Ellipsis, slice(None, 20, None))]
        query_pass_29 = query_states_88[(Ellipsis, slice(20, None, None))]
        query_states_88 = None
        key_rot_29 = key_states_88[(Ellipsis, slice(None, 20, None))]
        key_pass_29 = key_states_88[(Ellipsis, slice(20, None, None))]
        key_states_88 = None
        getitem_354 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_87 = getitem_354.to(dtype=torch.float16)
        getitem_354 = None
        getitem_355 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_87 = getitem_355.to(dtype=torch.float16)
        getitem_355 = None
        squeeze_116 = cos_87.squeeze(1)
        cos_87 = None
        cos_88 = squeeze_116.squeeze(0)
        squeeze_116 = None
        squeeze_118 = sin_87.squeeze(1)
        sin_87 = None
        sin_88 = squeeze_118.squeeze(0)
        squeeze_118 = None
        getitem_356 = cos_88[position_ids_1]
        cos_88 = None
        cos_89 = getitem_356.unsqueeze(1)
        getitem_356 = None
        getitem_357 = sin_88[position_ids_1]
        sin_88 = None
        sin_89 = getitem_357.unsqueeze(1)
        getitem_357 = None
        mul_145 = query_rot_29 * cos_89
        chunk_58 = torch.chunk(query_rot_29, 2, dim=-1)
        query_rot_29 = None
        x1_58 = chunk_58[0]
        x2_58 = chunk_58[1]
        chunk_58 = None
        neg_58 = -x2_58
        x2_58 = None
        cat_116 = torch.cat((neg_58, x1_58), dim=-1)
        neg_58 = x1_58 = None
        mul_146 = cat_116 * sin_89
        cat_116 = None
        q_embed_29 = mul_145 + mul_146
        mul_145 = mul_146 = None
        mul_147 = key_rot_29 * cos_89
        cos_89 = None
        chunk_59 = torch.chunk(key_rot_29, 2, dim=-1)
        key_rot_29 = None
        x1_59 = chunk_59[0]
        x2_59 = chunk_59[1]
        chunk_59 = None
        neg_59 = -x2_59
        x2_59 = None
        cat_117 = torch.cat((neg_59, x1_59), dim=-1)
        neg_59 = x1_59 = None
        mul_148 = cat_117 * sin_89
        cat_117 = sin_89 = None
        k_embed_29 = mul_147 + mul_148
        mul_147 = mul_148 = None
        query_states_89 = torch.cat((q_embed_29, query_pass_29), dim=-1)
        q_embed_29 = query_pass_29 = None
        key_states_89 = torch.cat((k_embed_29, key_pass_29), dim=-1)
        k_embed_29 = key_pass_29 = None
        transpose_148 = key_states_89.transpose(2, 3)
        key_states_89 = None
        matmul_58 = torch.matmul(query_states_89, transpose_148)
        query_states_89 = transpose_148 = None
        attn_weights_87 = matmul_58 / 8.94427190999916
        matmul_58 = None
        attn_weights_88 = attn_weights_87 + combined_attention_mask_1
        attn_weights_87 = None
        softmax_29 = torch.nn.functional.softmax(
            attn_weights_88, dim=-1, dtype=torch.float32
        )
        attn_weights_88 = None
        attn_weights_89 = softmax_29.to(torch.float16)
        softmax_29 = None
        attn_output_116 = torch.matmul(attn_weights_89, value_states_59)
        attn_weights_89 = value_states_59 = None
        transpose_149 = attn_output_116.transpose(1, 2)
        attn_output_116 = None
        attn_output_117 = transpose_149.contiguous()
        transpose_149 = None
        attn_output_118 = attn_output_117.reshape(1, 19, 2560)
        attn_output_117 = None
        attn_output_119 = torch._C._nn.linear(
            attn_output_118,
            l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_118 = l_self_modules_model_modules_layers_modules_29_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_146 = hidden_states_144 + attn_output_119
        hidden_states_144 = attn_output_119 = None
        hidden_states_147 = torch.nn.functional.layer_norm(
            hidden_states_146,
            (2560,),
            l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_29_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_207 = torch._C._nn.linear(
            hidden_states_147,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_29 = torch.nn.functional.silu(linear_207, inplace=False)
        linear_207 = None
        linear_208 = torch._C._nn.linear(
            hidden_states_147,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_147 = l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_149 = silu_29 * linear_208
        silu_29 = linear_208 = None
        hidden_states_148 = torch._C._nn.linear(
            mul_149,
            l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_149 = l_self_modules_model_modules_layers_modules_29_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_149 = hidden_states_146 + hidden_states_148
        hidden_states_146 = hidden_states_148 = None
        hidden_states_150 = torch.nn.functional.layer_norm(
            hidden_states_149,
            (2560,),
            l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_30_modules_input_layernorm_parameters_bias_ = (None)
        query_states_90 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_90 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_60 = torch._C._nn.linear(
            hidden_states_150,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_150 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_92 = query_states_90.view(1, 19, 32, 80)
        query_states_90 = None
        query_states_91 = view_92.transpose(1, 2)
        view_92 = None
        view_93 = key_states_90.view(1, 19, 32, 80)
        key_states_90 = None
        key_states_91 = view_93.transpose(1, 2)
        view_93 = None
        view_94 = value_states_60.view(1, 19, 32, 80)
        value_states_60 = None
        value_states_61 = view_94.transpose(1, 2)
        view_94 = None
        query_rot_30 = query_states_91[(Ellipsis, slice(None, 20, None))]
        query_pass_30 = query_states_91[(Ellipsis, slice(20, None, None))]
        query_states_91 = None
        key_rot_30 = key_states_91[(Ellipsis, slice(None, 20, None))]
        key_pass_30 = key_states_91[(Ellipsis, slice(20, None, None))]
        key_states_91 = None
        getitem_366 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_90 = getitem_366.to(dtype=torch.float16)
        getitem_366 = None
        getitem_367 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_90 = getitem_367.to(dtype=torch.float16)
        getitem_367 = None
        squeeze_120 = cos_90.squeeze(1)
        cos_90 = None
        cos_91 = squeeze_120.squeeze(0)
        squeeze_120 = None
        squeeze_122 = sin_90.squeeze(1)
        sin_90 = None
        sin_91 = squeeze_122.squeeze(0)
        squeeze_122 = None
        getitem_368 = cos_91[position_ids_1]
        cos_91 = None
        cos_92 = getitem_368.unsqueeze(1)
        getitem_368 = None
        getitem_369 = sin_91[position_ids_1]
        sin_91 = None
        sin_92 = getitem_369.unsqueeze(1)
        getitem_369 = None
        mul_150 = query_rot_30 * cos_92
        chunk_60 = torch.chunk(query_rot_30, 2, dim=-1)
        query_rot_30 = None
        x1_60 = chunk_60[0]
        x2_60 = chunk_60[1]
        chunk_60 = None
        neg_60 = -x2_60
        x2_60 = None
        cat_120 = torch.cat((neg_60, x1_60), dim=-1)
        neg_60 = x1_60 = None
        mul_151 = cat_120 * sin_92
        cat_120 = None
        q_embed_30 = mul_150 + mul_151
        mul_150 = mul_151 = None
        mul_152 = key_rot_30 * cos_92
        cos_92 = None
        chunk_61 = torch.chunk(key_rot_30, 2, dim=-1)
        key_rot_30 = None
        x1_61 = chunk_61[0]
        x2_61 = chunk_61[1]
        chunk_61 = None
        neg_61 = -x2_61
        x2_61 = None
        cat_121 = torch.cat((neg_61, x1_61), dim=-1)
        neg_61 = x1_61 = None
        mul_153 = cat_121 * sin_92
        cat_121 = sin_92 = None
        k_embed_30 = mul_152 + mul_153
        mul_152 = mul_153 = None
        query_states_92 = torch.cat((q_embed_30, query_pass_30), dim=-1)
        q_embed_30 = query_pass_30 = None
        key_states_92 = torch.cat((k_embed_30, key_pass_30), dim=-1)
        k_embed_30 = key_pass_30 = None
        transpose_153 = key_states_92.transpose(2, 3)
        key_states_92 = None
        matmul_60 = torch.matmul(query_states_92, transpose_153)
        query_states_92 = transpose_153 = None
        attn_weights_90 = matmul_60 / 8.94427190999916
        matmul_60 = None
        attn_weights_91 = attn_weights_90 + combined_attention_mask_1
        attn_weights_90 = None
        softmax_30 = torch.nn.functional.softmax(
            attn_weights_91, dim=-1, dtype=torch.float32
        )
        attn_weights_91 = None
        attn_weights_92 = softmax_30.to(torch.float16)
        softmax_30 = None
        attn_output_120 = torch.matmul(attn_weights_92, value_states_61)
        attn_weights_92 = value_states_61 = None
        transpose_154 = attn_output_120.transpose(1, 2)
        attn_output_120 = None
        attn_output_121 = transpose_154.contiguous()
        transpose_154 = None
        attn_output_122 = attn_output_121.reshape(1, 19, 2560)
        attn_output_121 = None
        attn_output_123 = torch._C._nn.linear(
            attn_output_122,
            l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_122 = l_self_modules_model_modules_layers_modules_30_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_151 = hidden_states_149 + attn_output_123
        hidden_states_149 = attn_output_123 = None
        hidden_states_152 = torch.nn.functional.layer_norm(
            hidden_states_151,
            (2560,),
            l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_30_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_214 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_30 = torch.nn.functional.silu(linear_214, inplace=False)
        linear_214 = None
        linear_215 = torch._C._nn.linear(
            hidden_states_152,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_152 = l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_154 = silu_30 * linear_215
        silu_30 = linear_215 = None
        hidden_states_153 = torch._C._nn.linear(
            mul_154,
            l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_154 = l_self_modules_model_modules_layers_modules_30_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_154 = hidden_states_151 + hidden_states_153
        hidden_states_151 = hidden_states_153 = None
        hidden_states_155 = torch.nn.functional.layer_norm(
            hidden_states_154,
            (2560,),
            l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_31_modules_input_layernorm_parameters_bias_ = (None)
        query_states_93 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_ = (
            None
        )
        key_states_93 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_ = (
            None
        )
        value_states_62 = torch._C._nn.linear(
            hidden_states_155,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_,
            None,
        )
        hidden_states_155 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_ = (None)
        view_95 = query_states_93.view(1, 19, 32, 80)
        query_states_93 = None
        query_states_94 = view_95.transpose(1, 2)
        view_95 = None
        view_96 = key_states_93.view(1, 19, 32, 80)
        key_states_93 = None
        key_states_94 = view_96.transpose(1, 2)
        view_96 = None
        view_97 = value_states_62.view(1, 19, 32, 80)
        value_states_62 = None
        value_states_63 = view_97.transpose(1, 2)
        view_97 = None
        query_rot_31 = query_states_94[(Ellipsis, slice(None, 20, None))]
        query_pass_31 = query_states_94[(Ellipsis, slice(20, None, None))]
        query_states_94 = None
        key_rot_31 = key_states_94[(Ellipsis, slice(None, 20, None))]
        key_pass_31 = key_states_94[(Ellipsis, slice(20, None, None))]
        key_states_94 = None
        getitem_378 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_cos_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_cos_cached_ = (
            None
        )
        cos_93 = getitem_378.to(dtype=torch.float16)
        getitem_378 = None
        getitem_379 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_sin_cached_[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 19, None),
                Ellipsis,
            )
        ]
        l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_rotary_emb_buffers_sin_cached_ = (
            None
        )
        sin_93 = getitem_379.to(dtype=torch.float16)
        getitem_379 = None
        squeeze_124 = cos_93.squeeze(1)
        cos_93 = None
        cos_94 = squeeze_124.squeeze(0)
        squeeze_124 = None
        squeeze_126 = sin_93.squeeze(1)
        sin_93 = None
        sin_94 = squeeze_126.squeeze(0)
        squeeze_126 = None
        getitem_380 = cos_94[position_ids_1]
        cos_94 = None
        cos_95 = getitem_380.unsqueeze(1)
        getitem_380 = None
        getitem_381 = sin_94[position_ids_1]
        sin_94 = position_ids_1 = None
        sin_95 = getitem_381.unsqueeze(1)
        getitem_381 = None
        mul_155 = query_rot_31 * cos_95
        chunk_62 = torch.chunk(query_rot_31, 2, dim=-1)
        query_rot_31 = None
        x1_62 = chunk_62[0]
        x2_62 = chunk_62[1]
        chunk_62 = None
        neg_62 = -x2_62
        x2_62 = None
        cat_124 = torch.cat((neg_62, x1_62), dim=-1)
        neg_62 = x1_62 = None
        mul_156 = cat_124 * sin_95
        cat_124 = None
        q_embed_31 = mul_155 + mul_156
        mul_155 = mul_156 = None
        mul_157 = key_rot_31 * cos_95
        cos_95 = None
        chunk_63 = torch.chunk(key_rot_31, 2, dim=-1)
        key_rot_31 = None
        x1_63 = chunk_63[0]
        x2_63 = chunk_63[1]
        chunk_63 = None
        neg_63 = -x2_63
        x2_63 = None
        cat_125 = torch.cat((neg_63, x1_63), dim=-1)
        neg_63 = x1_63 = None
        mul_158 = cat_125 * sin_95
        cat_125 = sin_95 = None
        k_embed_31 = mul_157 + mul_158
        mul_157 = mul_158 = None
        query_states_95 = torch.cat((q_embed_31, query_pass_31), dim=-1)
        q_embed_31 = query_pass_31 = None
        key_states_95 = torch.cat((k_embed_31, key_pass_31), dim=-1)
        k_embed_31 = key_pass_31 = None
        transpose_158 = key_states_95.transpose(2, 3)
        key_states_95 = None
        matmul_62 = torch.matmul(query_states_95, transpose_158)
        query_states_95 = transpose_158 = None
        attn_weights_93 = matmul_62 / 8.94427190999916
        matmul_62 = None
        attn_weights_94 = attn_weights_93 + combined_attention_mask_1
        attn_weights_93 = combined_attention_mask_1 = None
        softmax_31 = torch.nn.functional.softmax(
            attn_weights_94, dim=-1, dtype=torch.float32
        )
        attn_weights_94 = None
        attn_weights_95 = softmax_31.to(torch.float16)
        softmax_31 = None
        attn_output_124 = torch.matmul(attn_weights_95, value_states_63)
        attn_weights_95 = value_states_63 = None
        transpose_159 = attn_output_124.transpose(1, 2)
        attn_output_124 = None
        attn_output_125 = transpose_159.contiguous()
        transpose_159 = None
        attn_output_126 = attn_output_125.reshape(1, 19, 2560)
        attn_output_125 = None
        attn_output_127 = torch._C._nn.linear(
            attn_output_126,
            l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_,
            None,
        )
        attn_output_126 = l_self_modules_model_modules_layers_modules_31_modules_self_attn_modules_o_proj_parameters_weight_ = (None)
        hidden_states_156 = hidden_states_154 + attn_output_127
        hidden_states_154 = attn_output_127 = None
        hidden_states_157 = torch.nn.functional.layer_norm(
            hidden_states_156,
            (2560,),
            l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_,
            l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_bias_,
            1e-05,
        )
        l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_model_modules_layers_modules_31_modules_post_attention_layernorm_parameters_bias_ = (None)
        linear_221 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_,
            None,
        )
        l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_gate_proj_parameters_weight_ = (
            None
        )
        silu_31 = torch.nn.functional.silu(linear_221, inplace=False)
        linear_221 = None
        linear_222 = torch._C._nn.linear(
            hidden_states_157,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_,
            None,
        )
        hidden_states_157 = l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_up_proj_parameters_weight_ = (None)
        mul_159 = silu_31 * linear_222
        silu_31 = linear_222 = None
        hidden_states_158 = torch._C._nn.linear(
            mul_159,
            l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_,
            None,
        )
        mul_159 = l_self_modules_model_modules_layers_modules_31_modules_mlp_modules_down_proj_parameters_weight_ = (None)
        hidden_states_159 = hidden_states_156 + hidden_states_158
        hidden_states_156 = hidden_states_158 = None
        hidden_states_160 = torch.nn.functional.layer_norm(
            hidden_states_159,
            (2560,),
            l_self_modules_model_modules_norm_parameters_weight_,
            l_self_modules_model_modules_norm_parameters_bias_,
            1e-05,
        )
        hidden_states_159 = (
            l_self_modules_model_modules_norm_parameters_weight_
        ) = l_self_modules_model_modules_norm_parameters_bias_ = None
        linear_224 = torch._C._nn.linear(
            hidden_states_160, l_self_modules_lm_head_parameters_weight_, None
        )
        hidden_states_160 = l_self_modules_lm_head_parameters_weight_ = None
        logits = linear_224.float()
        linear_224 = None
        return (logits,)
